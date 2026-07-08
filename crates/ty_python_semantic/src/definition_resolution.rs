use std::collections::VecDeque;

use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use ty_module_resolver::Module;
use ty_python_core::definition::{
    Definition, DefinitionKind, DefinitionState, NestedBindingExecution,
};
use ty_python_core::scope::ScopeId;
use ty_python_core::{
    BindingWithConstraintsIterator, BoundnessAnalysis, Program, ProgramFile, global_scope,
    place_table, semantic_index, use_def_map,
};

use crate::place::{
    Place, RequiresExplicitReExport, builtins_module_scope, class_body_implicit_symbol,
    implicit_builtins_symbol, implicit_builtins_symbol_scope, is_reexported,
    module_type_implicit_global_symbol,
};
use crate::place_load::{
    ImplicitPlaceLoad, PlaceLoadResolution, PlaceLoadResolutionStep, PlaceLoadSource,
    PlaceLoadSourceKind,
};
use crate::reachability::ReachabilityConstraintsExtension;
use crate::types::ProgramEnvironment;
use crate::{Db, FxIndexSet};

/// Returns the source-backed definitions that may supply the value for a module
/// global at the end of its scope.
pub(crate) fn definitions_for_module_global<'db>(
    db: &'db dyn Db,
    program: Program<'db>,
    module: Module<'db>,
    name: &str,
) -> Option<DefinitionResolution<'db>> {
    let file = ProgramFile::new(db, module.file(db)?, program);
    let scope = global_scope(db, file);
    let symbol = place_table(db, scope).symbol_id(name)?;

    Some(DefinitionResolution::from_bindings(
        db,
        use_def_map(db, scope).end_of_scope_symbol_bindings(symbol),
        RequiresExplicitReExport::No,
    ))
}

/// Resolves the definitions for the ordered sources of a place load.
pub(crate) fn definitions_for_place_load<'db>(
    db: &'db dyn Db,
    environment: &ProgramEnvironment<'db>,
    scope: ScopeId<'db>,
    place_load: &mut PlaceLoadResolution<'db, '_>,
) -> DefinitionResolution<'db> {
    let mut resolution = DefinitionResolution {
        definitions: SmallVec::new(),
        is_complete: true,
        may_be_unbound: false,
        may_be_deleted: false,
    };
    let mut may_be_unbound = true;

    while may_be_unbound {
        let Some(step) = place_load.next() else {
            break;
        };
        match step {
            PlaceLoadResolutionStep::Source(source) => {
                let mut source_resolution =
                    DefinitionResolution::from_place_load_source(db, environment, scope, &source);
                if source.is_class_body_global_fallback() && source_resolution.has_value() {
                    source_resolution.may_be_unbound = false;
                }
                may_be_unbound = source_resolution.may_be_unbound;
                resolution.extend(source_resolution);
            }
            PlaceLoadResolutionStep::MemberResolutionCondition(_) => {
                resolution.is_complete = false;
                break;
            }
            PlaceLoadResolutionStep::Exhausted(_) => break,
        }
    }

    resolution.may_be_unbound = may_be_unbound;
    resolution
}

/// A set of definitions found by name resolution along with facts about their availability.
pub(crate) struct DefinitionResolution<'db> {
    definitions: SmallVec<[Definition<'db>; 2]>,
    is_complete: bool,
    may_be_unbound: bool,
    may_be_deleted: bool,
}

impl<'db> DefinitionResolution<'db> {
    /// Returns the definitions found by name resolution.
    pub(crate) fn definitions(&self) -> &[Definition<'db>] {
        &self.definitions
    }

    /// Returns whether every possible result is represented by a definition.
    pub(crate) fn is_complete(&self) -> bool {
        self.is_complete
    }

    /// Returns whether the value is bound on every reachable control-flow path.
    pub(crate) fn is_definitely_bound(&self) -> bool {
        !self.may_be_unbound
    }

    /// Returns whether a reachable deletion may leave the value unbound.
    pub(crate) fn may_be_deleted(&self) -> bool {
        self.may_be_deleted
    }

    fn from_place_load_source(
        db: &'db dyn Db,
        environment: &ProgramEnvironment<'db>,
        scope: ScopeId<'db>,
        source: &PlaceLoadSource<'db>,
    ) -> Self {
        match &source.kind {
            PlaceLoadSourceKind::Bindings(bindings) => {
                Self::from_bindings(db, bindings.clone(), RequiresExplicitReExport::No)
            }
            PlaceLoadSourceKind::DefinitionsFromOwningScope { scope, id } => Self::from_bindings(
                db,
                use_def_map(db, *scope).reachable_bindings(*id),
                RequiresExplicitReExport::No,
            ),
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::ExplicitGlobalSymbol {
                file,
                name,
            }) => {
                let scope = global_scope(db, *file);
                let Some(symbol) = place_table(db, scope).symbol_id(name) else {
                    return Self {
                        definitions: SmallVec::new(),
                        is_complete: true,
                        may_be_unbound: true,
                        may_be_deleted: false,
                    };
                };
                Self::from_bindings(
                    db,
                    use_def_map(db, scope).reachable_symbol_bindings(symbol),
                    RequiresExplicitReExport::No,
                )
            }
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::DunderClass(class_def)) => {
                let mut resolution = Self {
                    definitions: SmallVec::new(),
                    is_complete: true,
                    may_be_unbound: false,
                    may_be_deleted: false,
                };
                resolution.push_definition(db, *class_def);
                resolution
            }
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::ClassBodySymbol(name)) => {
                Self::from_place_without_definition(
                    class_body_implicit_symbol(db, environment, name).place,
                )
            }
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::ModuleImplicitGlobal {
                file,
                name,
            }) => Self::from_place_without_definition(
                module_type_implicit_global_symbol(db, *file, name).place,
            ),
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::Builtin(name)) => {
                Self::from_builtin(db, environment, scope, name)
            }
        }
    }

    fn from_builtin(
        db: &'db dyn Db,
        environment: &ProgramEnvironment<'db>,
        scope: ScopeId<'db>,
        name: &str,
    ) -> Self {
        if Some(scope) == builtins_module_scope(db, environment) {
            // A missing name in `builtins` cannot fall back to the module that is currently being
            // resolved. Treating it as undefined also avoids a recursive semantic query.
            return Self::from_place_without_definition(Place::Undefined);
        }

        let Some(builtins_scope) = implicit_builtins_symbol_scope(db, environment, name) else {
            // No runtime-visible builtin supplies this name.
            return Self::from_place_without_definition(Place::Undefined);
        };

        if builtins_scope == scope {
            // End-of-scope lookup in a project-level `__builtins__` module can select a binding
            // that occurs after this load. Keep its inferred value unrepresented instead of
            // exposing that later binding as the load's source definition.
            return Self::from_place_without_definition(
                implicit_builtins_symbol(db, environment, name).place,
            );
        }

        let Some(symbol) = place_table(db, builtins_scope).symbol_id(name) else {
            // The module supplies this name through a synthetic module attribute rather than an
            // explicit symbol, so there is no source definition to expose.
            return Self::from_place_without_definition(
                implicit_builtins_symbol(db, environment, name).place,
            );
        };

        let mut resolution = Self::from_bindings(
            db,
            use_def_map(db, builtins_scope).end_of_scope_symbol_bindings(symbol),
            RequiresExplicitReExport::Yes,
        );

        // The target definitions are useful for navigation, but the implicit fallback that
        // connects this load to them has no source representation that a refactor can safely
        // rewrite.
        resolution.is_complete = false;

        resolution
    }

    fn from_bindings(
        db: &'db dyn Db,
        mut bindings: BindingWithConstraintsIterator<'db, 'db>,
        requires_explicit_reexport: RequiresExplicitReExport,
    ) -> Self {
        let boundness = bindings.boundness_analysis();
        let mut resolution = Self {
            definitions: SmallVec::new(),
            is_complete: true,
            may_be_unbound: false,
            may_be_deleted: false,
        };
        let mut has_defined_binding = false;

        while let Some(binding) = bindings.next() {
            let reachability = bindings.reachability_constraints().evaluate(
                db,
                bindings.predicates(),
                binding.reachability_constraint,
            );
            if reachability.is_always_false() {
                continue;
            }

            match binding.binding {
                DefinitionState::Defined(definition)
                    if matches!(requires_explicit_reexport, RequiresExplicitReExport::Yes)
                        && !is_reexported(db, definition) =>
                {
                    resolution.may_be_unbound |= reachability.may_be_true();
                }
                DefinitionState::Defined(definition) => {
                    has_defined_binding = true;
                    resolution.push_definition(db, definition);
                }
                DefinitionState::Deleted => {
                    let may_be_deleted = reachability.may_be_true();
                    resolution.may_be_unbound |= may_be_deleted;
                    resolution.may_be_deleted |= may_be_deleted;
                }
                DefinitionState::Undefined
                    if boundness == BoundnessAnalysis::BasedOnUnboundVisibility =>
                {
                    resolution.may_be_unbound |= reachability.may_be_true();
                }
                DefinitionState::Undefined => {}
            }
        }

        if !has_defined_binding {
            resolution.may_be_unbound = true;
        }

        resolution
    }

    fn push_definition(&mut self, db: &'db dyn Db, definition: Definition<'db>) {
        let definitions = source_backed_definitions(db, [definition]);
        if definitions.is_empty() {
            self.is_complete = false;
            return;
        }

        for definition in definitions {
            if !self.definitions.contains(&definition) {
                self.definitions.push(definition);
            }
        }
    }

    fn from_place_without_definition(place: Place<'db>) -> Self {
        Self {
            definitions: SmallVec::new(),
            is_complete: place.is_undefined(),
            may_be_unbound: !place.is_definitely_bound(),
            may_be_deleted: false,
        }
    }

    /// Returns whether resolution found a value, including one without a source definition.
    fn has_value(&self) -> bool {
        !self.definitions.is_empty() || !self.is_complete
    }

    fn extend(&mut self, other: Self) {
        for definition in other.definitions {
            if !self.definitions.contains(&definition) {
                self.definitions.push(definition);
            }
        }
        self.is_complete &= other.is_complete;
        self.may_be_deleted |= other.may_be_deleted;
    }
}

/// Returns definitions that are represented by a source location that a user
/// can actually edit.
///
/// This result excludes synthetic definitions like loop headers or nested bindings.
///
/// Comprehension walruses are represented in the containing scope by synthetic eager bindings:
///
/// ```python
/// [(last := item) for item in items]
/// print(last)  # Go to definition should select `last := item` above.
/// ```
///
/// The binding for the use in `print` is synthetic, so this follows it into the comprehension's
/// end-of-scope bindings. Nested comprehensions can produce a chain of these proxies. It only
/// follows sources that resolve to the same variable, so `global` and `nonlocal` writes do not
/// become definitions of each other.
pub(crate) fn source_backed_definitions<'db>(
    db: &'db dyn Db,
    definitions: impl IntoIterator<Item = Definition<'db>>,
) -> FxIndexSet<Definition<'db>> {
    let mut pending = definitions.into_iter().collect::<VecDeque<_>>();
    let mut seen = FxHashSet::default();
    let mut result = FxIndexSet::default();

    while let Some(definition) = pending.pop_front() {
        if !seen.insert(definition) {
            continue;
        }

        match definition.kind(db) {
            DefinitionKind::NestedBindings(nested) => {
                let index = semantic_index(db, definition.program_file(db));
                let sources = nested
                    .visible_binding_sources(index, definition.file_scope(db))
                    .flatten()
                    .filter_map(|binding| binding.binding.definition());
                // A lazy function proxy can lead to an eager comprehension proxy. Follow that
                // proxy-only chain without exposing ordinary lazy nested assignments.
                pending.extend(sources.filter(|source| {
                    nested.execution == NestedBindingExecution::Eager
                        || matches!(source.kind(db), DefinitionKind::NestedBindings(_))
                }));
            }
            kind if kind.is_user_visible() => {
                result.insert(definition);
            }
            _ => {}
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use ruff_db::files::system_path_to_file;
    use ty_python_core::ProgramFile;

    use super::definitions_for_module_global;
    use crate::SemanticModel;
    use crate::db::tests::TestDbBuilder;

    #[test]
    fn definitions_for_module_global_retains_conditional_definitions() {
        let db = TestDbBuilder::new()
            .with_file(
                "/src/pkg/__init__.py",
                r#"
if flag:
    from . import first as value
else:
    from . import second as value
"#,
            )
            .with_file("/src/pkg/first.py", "")
            .with_file("/src/pkg/second.py", "")
            .with_file("/src/use.py", "import pkg")
            .build()
            .expect("valid TestDb setup");
        let file = system_path_to_file(&db, "/src/use.py").expect("test file should exist");
        let program = db.program_environment().program(&db);
        let model = SemanticModel::new(&db, ProgramFile::new(&db, file, program));
        let module = model
            .resolve_module(Some("pkg"), 0)
            .expect("test package should resolve");

        let resolution = definitions_for_module_global(&db, program, module, "value")
            .expect("module global should exist");

        assert_eq!(resolution.definitions().len(), 2);
        assert!(resolution.is_complete());
        assert!(resolution.is_definitely_bound());
    }
}

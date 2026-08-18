use std::collections::VecDeque;

use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use ty_module_resolver::Module;
use ty_python_core::definition::{
    Definition, DefinitionKind, DefinitionState, NestedBindingExecution,
};
use ty_python_core::{
    BindingWithConstraintsIterator, BoundnessAnalysis, Program, ProgramFile, global_scope,
    place_table, semantic_index, use_def_map,
};

use crate::reachability::ReachabilityConstraintsExtension;
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
    ))
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

    /// Returns whether every resolved binding is represented by a source definition.
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

    fn from_bindings(
        db: &'db dyn Db,
        mut bindings: BindingWithConstraintsIterator<'db, 'db>,
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

//! This module exposes name resolution and reaching-definition analysis to
//! the language server without requiring it to run type inference.
//!
//! It provides two entry points on [`SemanticModel`]:
//!
//! - [`SemanticModel::name_load`] resolves a name load in the model's file. It selects the
//!   appropriate point-in-time, deferred, or string-annotation binding state and returns
//!   [`NameLoadResolution`]. It returns `None` when selecting that state requires type inference.
//! - [`SemanticModel::module_global_providers`] resolves an explicit module global using the
//!   bindings available at the end of the module. For example, a consumer analyzing
//!   `service.handler(request)` can resolve `service` to a module and then query that module for
//!   the member name `"handler"`. Unlike [`SemanticModel::name_load`], this does not require an
//!   `ExprName` load such as the `handler` in a direct `handler(request)` call.
//!
//! Both entry points expose [`ValueProviders`], which describes the direct source definitions
//! that may provide a value and whether the value may be unbound, deleted, or supplied by
//! something without a source definition. [`NameLoadResolution`] additionally reports whether
//! resolution crosses a `global` or `nonlocal` declaration. Providers are deliberately not
//! followed recursively: consumers can use them as edges between a load and its possible
//! definitions without also selecting sibling loads of those definitions.
//!
//! ## Example
//!
//! ```py
//! if use_fallback:
//!     from .fallback import handler  # definition A
//! else:
//!     from .primary import handler  # definition B
//!
//! handler(request)  # load U
//! ```
//!
//! Calling [`SemanticModel::name_load`] for `U` returns a [`NameLoadResolution`] whose providers
//! contain definitions A and B. Because every branch defines `handler`,
//! [`ValueProviders::is_definitely_bound`] returns `true`. If the `else` branch were absent, the
//! providers would still contain definition A, but `is_definitely_bound` would return `false`.
//!
//! Calling [`SemanticModel::module_global_providers`] for `handler` in this module returns the same
//! direct definitions from the module's end-of-scope binding state.

use ruff_python_ast as ast;
use smallvec::SmallVec;
use ty_module_resolver::Module;
use ty_python_core::definition::{Definition, DefinitionState};
use ty_python_core::place::PlaceExpr;
use ty_python_core::scope::ScopeId;
use ty_python_core::{
    BindingWithConstraintsIterator, BoundnessAnalysis, ProgramFile, global_scope, place_table,
    semantic_index, use_def_map,
};

use crate::place::{
    Place, RequiresExplicitReExport, builtins_module_scope, class_body_implicit_symbol,
    implicit_builtins_symbol, implicit_builtins_symbol_scope, is_reexported,
    module_type_implicit_global_symbol,
};
use crate::place_load::{
    ImplicitPlaceLoad, PlaceLoadMode, PlaceLoadResolution, PlaceLoadResolutionStep,
    PlaceLoadSource, PlaceLoadSourceKind, resolve_place_load,
};
use crate::reachability::ReachabilityConstraintsExtension;
use crate::types::ProgramEnvironment;
use crate::{Db, SemanticModel};

use super::user_visible_definitions;

impl<'db> SemanticModel<'db> {
    /// Resolves the possible value providers for a name load.
    ///
    /// Returns `None` when choosing the correct binding state would require type inference.
    pub fn name_load(&self, name: &ast::ExprName) -> Option<NameLoadResolution<'db>> {
        let environment = self.program_environment();
        let index = semantic_index(self.db(), self.program_file());
        let scope = self
            .scope(name.into())?
            .to_scope_id(self.db(), self.program_file());
        let mode = if self.is_in_string_annotation() {
            PlaceLoadMode::StringAnnotation
        } else if index.place_load_is_deferred(ast::ExprRef::Name(name))? {
            PlaceLoadMode::Deferred
        } else {
            PlaceLoadMode::AtExpression(name.into())
        };

        let place_load_resolution = resolve_place_load(
            self.db(),
            index,
            scope,
            PlaceExpr::from_expr_name(name),
            mode,
        );

        Some(NameLoadResolution::from_place_load_resolution(
            self.db(),
            &environment,
            scope,
            place_load_resolution,
        ))
    }

    /// Returns the possible providers for an explicit module global.
    pub fn module_global_providers(
        &self,
        module: Module<'db>,
        name: &str,
    ) -> Option<ValueProviders<'db>> {
        let file = ProgramFile::new(self.db(), module.file(self.db())?, self.program());
        let scope = global_scope(self.db(), file);
        let symbol = place_table(self.db(), scope).symbol_id(name)?;

        Some(ValueProviders::from_bindings(
            self.db(),
            use_def_map(self.db(), scope).end_of_scope_symbol_bindings(symbol),
            RequiresExplicitReExport::No,
        ))
    }
}

/// The direct value providers for a name load.
pub struct NameLoadResolution<'db> {
    providers: ValueProviders<'db>,
    crosses_scope_declaration: bool,
}

impl<'db> NameLoadResolution<'db> {
    /// Returns the possible value providers.
    pub fn providers(&self) -> &ValueProviders<'db> {
        &self.providers
    }

    /// Returns whether resolution crosses a `global` or `nonlocal` declaration.
    pub fn crosses_scope_declaration(&self) -> bool {
        self.crosses_scope_declaration
    }

    fn from_place_load_resolution(
        db: &'db dyn Db,
        environment: &ProgramEnvironment<'db>,
        scope: ScopeId<'db>,
        mut place_load_resolution: PlaceLoadResolution<'db, '_>,
    ) -> Self {
        let mut providers = ValueProviders::default();
        let mut may_be_unbound = true;

        while may_be_unbound {
            let Some(step) = place_load_resolution.next() else {
                break;
            };
            match step {
                PlaceLoadResolutionStep::Source(source) => {
                    let mut source_providers =
                        ValueProviders::from_source(db, environment, scope, &source);
                    if source.is_class_body_global_fallback() && source_providers.has_provider() {
                        source_providers.may_be_unbound = false;
                    }
                    may_be_unbound = source_providers.may_be_unbound;
                    providers.extend(source_providers);
                }
                PlaceLoadResolutionStep::MemberResolutionCondition(_) => {
                    providers.has_unrepresented_provider = true;
                    break;
                }
                PlaceLoadResolutionStep::Exhausted(_) => break,
            }
        }

        providers.may_be_unbound = may_be_unbound;

        Self {
            providers,
            crosses_scope_declaration: place_load_resolution.crosses_scope_declaration(),
        }
    }
}

/// The direct value providers for one load or an exported module global.
///
/// This does not recursively follow the providers' own inputs. Consumers can therefore use these
/// definitions as edges between a load and the bindings that may supply it without also selecting
/// sibling loads. [`Self::has_unrepresented_provider`] can be true even when definitions are
/// available: an implicit builtin load, for example, has a definition target but no import or
/// binding at the load site that a refactor can rewrite.
pub struct ValueProviders<'db> {
    definitions: SmallVec<[Definition<'db>; 2]>,
    has_unrepresented_provider: bool,
    may_be_unbound: bool,
    may_be_deleted: bool,
}

impl<'db> ValueProviders<'db> {
    /// Returns the source definitions that directly provide the value.
    pub fn definitions(&self) -> impl ExactSizeIterator<Item = Definition<'db>> + '_ {
        self.definitions.iter().copied()
    }

    /// Returns whether at least one provider relationship has no source representation.
    pub fn has_unrepresented_provider(&self) -> bool {
        self.has_unrepresented_provider
    }

    /// Returns whether every feasible path provides a value.
    pub fn is_definitely_bound(&self) -> bool {
        !self.may_be_unbound
    }

    /// Returns whether a reachable deletion may leave the value unbound.
    pub fn may_be_deleted(&self) -> bool {
        self.may_be_deleted
    }

    fn from_source(
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
                        may_be_unbound: true,
                        ..Self::default()
                    };
                };
                Self::from_bindings(
                    db,
                    use_def_map(db, scope).reachable_symbol_bindings(symbol),
                    RequiresExplicitReExport::No,
                )
            }
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::DunderClass(class_def)) => {
                let mut providers = Self::default();
                providers.push_definition(db, *class_def);
                providers
            }
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::ClassBodySymbol(name)) => {
                Self::from_unrepresented_source(
                    class_body_implicit_symbol(db, environment, name).place,
                )
            }
            PlaceLoadSourceKind::Implicit(ImplicitPlaceLoad::ModuleImplicitGlobal {
                file,
                name,
            }) => Self::from_unrepresented_source(
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
            return Self::from_unrepresented_source(Place::Undefined);
        }

        let Some(builtins_scope) = implicit_builtins_symbol_scope(db, environment, name) else {
            // No runtime-visible builtin supplies this name.
            return Self::from_unrepresented_source(Place::Undefined);
        };

        if builtins_scope == scope {
            // End-of-scope lookup in a project-level `__builtins__` module can select a binding
            // that occurs after this load. Keep its inferred value unrepresented instead of
            // exposing that later binding as the load's source definition.
            return Self::from_unrepresented_source(
                implicit_builtins_symbol(db, environment, name).place,
            );
        }

        let Some(symbol) = place_table(db, builtins_scope).symbol_id(name) else {
            // The module supplies this name through a synthetic module attribute rather than an
            // explicit symbol, so there is no source definition to expose.
            return Self::from_unrepresented_source(
                implicit_builtins_symbol(db, environment, name).place,
            );
        };

        let mut providers = Self::from_bindings(
            db,
            use_def_map(db, builtins_scope).end_of_scope_symbol_bindings(symbol),
            RequiresExplicitReExport::Yes,
        );

        // The target definitions are useful for navigation, but the implicit fallback that
        // connects this load to them has no source representation that a refactor can safely
        // rewrite.
        providers.has_unrepresented_provider = true;

        providers
    }

    fn from_bindings(
        db: &'db dyn Db,
        mut bindings: BindingWithConstraintsIterator<'db, 'db>,
        requires_explicit_reexport: RequiresExplicitReExport,
    ) -> Self {
        let boundness = bindings.boundness_analysis();
        let mut providers = Self::default();

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
                    providers.may_be_unbound |= reachability.may_be_true();
                }
                DefinitionState::Defined(definition) => providers.push_definition(db, definition),
                DefinitionState::Deleted => {
                    let may_be_deleted = reachability.may_be_true();
                    providers.may_be_unbound |= may_be_deleted;
                    providers.may_be_deleted |= may_be_deleted;
                }
                DefinitionState::Undefined
                    if boundness == BoundnessAnalysis::BasedOnUnboundVisibility =>
                {
                    providers.may_be_unbound |= reachability.may_be_true();
                }
                DefinitionState::Undefined => {}
            }
        }

        if !providers.has_provider() {
            providers.may_be_unbound = true;
        }
        providers
    }

    fn push_definition(&mut self, db: &'db dyn Db, definition: Definition<'db>) {
        let definitions = user_visible_definitions(db, [definition]);
        if definitions.is_empty() {
            self.has_unrepresented_provider = true;
            return;
        }
        for definition in definitions {
            if !self.definitions.contains(&definition) {
                self.definitions.push(definition);
            }
        }
    }

    fn from_unrepresented_source(place: Place<'db>) -> Self {
        Self {
            has_unrepresented_provider: !place.is_undefined(),
            may_be_unbound: !place.is_definitely_bound(),
            ..Self::default()
        }
    }

    fn has_provider(&self) -> bool {
        self.has_unrepresented_provider || !self.definitions.is_empty()
    }

    fn extend(&mut self, other: Self) {
        for definition in other.definitions {
            if !self.definitions.contains(&definition) {
                self.definitions.push(definition);
            }
        }
        self.has_unrepresented_provider |= other.has_unrepresented_provider;
        self.may_be_deleted |= other.may_be_deleted;
    }
}

impl Default for ValueProviders<'_> {
    fn default() -> Self {
        Self {
            definitions: SmallVec::new(),
            has_unrepresented_provider: false,
            may_be_unbound: false,
            may_be_deleted: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use ruff_db::files::system_path_to_file;
    use ruff_db::parsed::{ParsedModuleRef, parsed_module};
    use ruff_db::source::{SourceText, source_text};
    use ruff_python_ast::visitor::{Visitor, walk_expr};
    use ruff_python_ast::{self as ast, PythonVersion};
    use ruff_text_size::Ranged;
    use ty_python_core::ProgramFile;

    use super::NameLoadResolution;
    use crate::SemanticModel;
    use crate::db::tests::{TestDb, TestDbBuilder};

    #[test]
    fn name_load_uses_point_in_time_bindings() {
        let db = test_db(
            r#"
import first as value
before = value
import second as value
"#,
        );
        let test = NameResolutionTest::new(&db, PYTHON_FILE);
        let load = test.name_load("value");

        assert_eq!(test.provider_texts(&load), ["first as value"]);
    }

    #[test]
    fn name_load_uses_end_of_scope_bindings_for_deferred_annotations() {
        let db = stub_test_db(
            r#"
import first as value
before: value.C
import second as value
"#,
        );
        let test = NameResolutionTest::new(&db, STUB_FILE);
        let load = test.name_load("value");

        assert_eq!(
            test.provider_texts(&load),
            ["first as value", "second as value"]
        );
    }

    #[test]
    fn name_load_uses_end_of_scope_bindings_with_future_annotations() {
        let db = build_test_db(
            TestDbBuilder::new()
                .with_python_version(PythonVersion::PY313)
                .with_file(
                    PYTHON_FILE,
                    r#"
from __future__ import annotations
import first as value
before: value.C
import second as value
"#,
                ),
        );
        let test = NameResolutionTest::new(&db, PYTHON_FILE);
        let load = test.name_load("value");

        assert_eq!(
            test.provider_texts(&load),
            ["first as value", "second as value"]
        );
    }

    #[test]
    fn name_load_uses_end_of_scope_bindings_for_python_314_annotations() {
        let db = build_test_db(
            TestDbBuilder::new()
                .with_python_version(PythonVersion::PY314)
                .with_file(
                    PYTHON_FILE,
                    r#"
import first as value
before: value.C
import second as value
"#,
                ),
        );
        let test = NameResolutionTest::new(&db, PYTHON_FILE);
        let load = test.name_load("value");

        assert_eq!(
            test.provider_texts(&load),
            ["first as value", "second as value"]
        );
    }

    #[test]
    fn name_load_returns_none_when_deferredness_requires_inference() {
        let db = stub_test_db(
            r#"
import first as value
before = value
"#,
        );
        let test = NameResolutionTest::new(&db, STUB_FILE);

        assert!(test.try_name_load("value").is_none());
    }

    #[test]
    fn name_load_uses_end_of_scope_bindings_in_other_deferred_contexts() {
        let db = stub_test_db(
            r#"
import first as value
def function[T: value.C](arg=value): ...
class Class(value.C): ...
type Alias = value.C
callback = lambda arg=value: None
import second as value
"#,
        );
        let test = NameResolutionTest::new(&db, STUB_FILE);
        let loads = test.name_loads("value");

        assert_eq!(loads.len(), 5);
        for load in loads {
            assert_eq!(
                test.provider_texts(&load),
                ["first as value", "second as value"]
            );
        }
    }

    #[test]
    fn name_load_excludes_bindings_that_do_not_reach_the_use() {
        let db = test_db(
            r#"
def test(flag: bool):
    if flag:
        x: int = 1
        return
    x = 2
    print(x)
"#,
        );
        let test = NameResolutionTest::new(&db, PYTHON_FILE);
        let load = test.name_load("x");

        assert_eq!(test.provider_texts(&load), ["x = 2"]);
    }

    #[test]
    fn name_load_respects_redeclarations() {
        let db = test_db(
            r#"
def test(flag: bool):
    if flag:
        x: int = 10
    else:
        x: str = 'test'
    print(x)
    x: int = 30
    print(x)
"#,
        );
        let test = NameResolutionTest::new(&db, PYTHON_FILE);
        let loads = test.name_loads("x");
        let [first_load, second_load] = loads.as_slice() else {
            panic!("expected two loads of `x`");
        };

        assert_eq!(
            test.provider_texts(first_load),
            ["x: int = 10", "x: str = 'test'"]
        );
        assert_eq!(test.provider_texts(second_load), ["x: int = 30"]);
    }

    #[test]
    fn name_load_reports_possible_unboundness() {
        let db = build_test_db(
            TestDbBuilder::new()
                .with_file(
                    PYTHON_FILE,
                    r#"
def test(flag: bool):
    if flag:
        import other as value
    return value
"#,
                )
                .with_file(
                    "/src/other.py",
                    r#"
"#,
                ),
        );
        let test = NameResolutionTest::new(&db, PYTHON_FILE);
        let load = test.name_load("value");

        assert_eq!(test.provider_texts(&load), ["other as value"]);
        assert!(!load.providers().is_definitely_bound());
        assert!(!load.providers().may_be_deleted());
    }

    #[test]
    fn name_load_reports_reachable_deletions() {
        let db = build_test_db(
            TestDbBuilder::new()
                .with_file(
                    PYTHON_FILE,
                    r#"
def test(flag: bool):
    import other as value
    if flag:
        del value
    return value
"#,
                )
                .with_file(
                    "/src/other.py",
                    r#"
"#,
                ),
        );
        let test = NameResolutionTest::new(&db, PYTHON_FILE);
        let load = test.name_load("value");

        assert_eq!(test.provider_texts(&load), ["other as value"]);
        assert!(!load.providers().is_definitely_bound());
        assert!(load.providers().may_be_deleted());
    }

    #[test]
    fn name_load_distinguishes_implicit_providers_from_missing_names() {
        let db = test_db(
            r#"
def test():
    return int, missing_name
"#,
        );
        let test = NameResolutionTest::new(&db, PYTHON_FILE);
        let builtin = test.name_load("int");
        let missing = test.name_load("missing_name");

        let builtin_definitions = builtin.providers().definitions().collect::<Vec<_>>();
        let [builtin_definition] = builtin_definitions.as_slice() else {
            panic!("expected exactly one provider for `int`");
        };
        assert_eq!(builtin_definition.name(&db).as_deref(), Some("int"));
        assert!(builtin.providers().has_unrepresented_provider());
        assert!(builtin.providers().is_definitely_bound());
        assert!(!missing.providers().has_unrepresented_provider());
        assert!(!missing.providers().is_definitely_bound());
    }

    #[test]
    fn implicit_builtin_providers_require_explicit_reexports() {
        let db = build_test_db(
            TestDbBuilder::new()
                .with_file(
                    "/src/__builtins__.pyi",
                    r#"
flag: bool
if flag:
    from first import value as value
else:
    from second import value
"#,
                )
                .with_file(
                    "/src/first.py",
                    r#"
value = 1
"#,
                )
                .with_file(
                    "/src/second.py",
                    r#"
value = 2
"#,
                )
                .with_file(
                    PYTHON_FILE,
                    r#"
result = value
"#,
                ),
        );
        let test = NameResolutionTest::new(&db, PYTHON_FILE);
        let load = test.name_load("value");

        assert_eq!(load.providers().definitions().count(), 1);
        assert!(load.providers().has_unrepresented_provider());
        assert!(!load.providers().is_definitely_bound());
    }

    #[test]
    fn implicit_builtin_providers_do_not_use_later_bindings_from_the_same_scope() {
        let db = build_test_db(
            TestDbBuilder::new()
                .with_file(
                    "/src/__builtins__.py",
                    r#"
before = value
from first import value as value
"#,
                )
                .with_file(
                    "/src/first.py",
                    r#"
value = 1
"#,
                ),
        );
        let test = NameResolutionTest::new(&db, "/src/__builtins__.py");
        let load = test.name_load("value");

        assert_eq!(load.providers().definitions().count(), 0);
        assert!(load.providers().has_unrepresented_provider());
    }

    #[test]
    fn name_load_reports_scope_declarations() {
        let db = test_db(
            r#"
value = 1
def test():
    global value
    return value
"#,
        );
        let test = NameResolutionTest::new(&db, PYTHON_FILE);
        let load = test.name_load("value");

        assert_eq!(test.provider_texts(&load), ["value = 1"]);
        assert!(load.crosses_scope_declaration());
    }

    #[test]
    fn module_global_providers_use_end_of_scope_bindings() {
        let db = build_test_db(
            TestDbBuilder::new()
                .with_file(
                    "/src/pkg/__init__.py",
                    r#"
if flag:
    from . import first as value
else:
    from . import second as value
"#,
                )
                .with_file(
                    "/src/pkg/first.py",
                    r#"
"#,
                )
                .with_file(
                    "/src/pkg/second.py",
                    r#"
"#,
                )
                .with_file(
                    "/src/use.py",
                    r#"
import pkg
"#,
                ),
        );
        let test = NameResolutionTest::new(&db, "/src/use.py");
        let module = test.model.resolve_module(Some("pkg"), 0).unwrap();
        let providers = test.model.module_global_providers(module, "value").unwrap();

        assert_eq!(providers.definitions().count(), 2);
        assert!(providers.is_definitely_bound());
    }

    const PYTHON_FILE: &str = "/src/foo.py";
    const STUB_FILE: &str = "/src/foo.pyi";

    fn test_db(source: &str) -> TestDb {
        build_test_db(TestDbBuilder::new().with_file(PYTHON_FILE, source))
    }

    fn stub_test_db(source: &str) -> TestDb {
        build_test_db(TestDbBuilder::new().with_file(STUB_FILE, source))
    }

    fn build_test_db(builder: TestDbBuilder<'_>) -> TestDb {
        builder.build().expect("valid TestDb setup")
    }

    struct NameResolutionTest<'db> {
        db: &'db TestDb,
        model: SemanticModel<'db>,
        module: ParsedModuleRef,
        source: SourceText,
    }

    impl<'db> NameResolutionTest<'db> {
        fn new(db: &'db TestDb, path: &str) -> Self {
            let Ok(file) = system_path_to_file(db, path) else {
                panic!("test file `{path}` should exist");
            };
            let file = ProgramFile::new(db, file, db.program_environment().program(db));
            let module = parsed_module(db, file.python_file(db)).load(db);
            let source = source_text(db, file.file(db));
            let model = SemanticModel::new(db, file);
            Self {
                db,
                model,
                module,
                source,
            }
        }

        fn try_name_load(&self, search: &str) -> Option<NameLoadResolution<'db>> {
            let names = loaded_names(self.module.syntax(), search);
            let &[name] = names.as_slice() else {
                panic!(
                    "expected exactly one load of `{search}`, found {}",
                    names.len()
                );
            };
            self.model.name_load(name)
        }

        fn name_load(&self, search: &str) -> NameLoadResolution<'db> {
            self.try_name_load(search)
                .unwrap_or_else(|| panic!("expected `{search}` load to be supported"))
        }

        fn name_loads(&self, searches: &str) -> Vec<NameLoadResolution<'db>> {
            loaded_names(self.module.syntax(), searches)
                .into_iter()
                .map(|name| {
                    self.model.name_load(name).unwrap_or_else(|| {
                        panic!("expected every `{searches}` load to be supported")
                    })
                })
                .collect()
        }

        fn provider_texts<'a>(&'a self, load: &NameLoadResolution<'db>) -> Vec<&'a str> {
            load.providers()
                .definitions()
                .map(|definition| {
                    let range = definition.full_range(self.db, &self.module).range();
                    &self.source[range]
                })
                .collect()
        }
    }

    fn loaded_names<'ast>(
        module: &'ast ast::ModModule,
        searched: &str,
    ) -> Vec<&'ast ast::ExprName> {
        struct Collector<'ast, 'name> {
            searched: &'name str,
            names: Vec<&'ast ast::ExprName>,
        }

        impl<'ast> Visitor<'ast> for Collector<'ast, '_> {
            fn visit_expr(&mut self, expression: &'ast ast::Expr) {
                if let ast::Expr::Name(name) = expression
                    && name.ctx.is_load()
                    && name.id == self.searched
                {
                    self.names.push(name);
                }
                walk_expr(self, expression);
            }
        }

        let mut collector = Collector {
            searched,
            names: Vec::new(),
        };
        collector.visit_body(&module.body);
        collector.names
    }
}

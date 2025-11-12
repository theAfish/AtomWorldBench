"""Test registry.py functionality with abstract base & non-abstract subclass rules."""

import pytest
from types import ModuleType
from unittest.mock import patch, MagicMock
from abc import ABC, abstractmethod

from AtomWorldBench.common.registry import (
    aliases_from_class_name,
    register,
    get_registered,
    derived_class_factory,
    load_plugins,
    _REGISTRY
)


class MockBase(ABC):
    """Abstract base class for testing."""

    @abstractmethod
    def foo(self):
        pass


def setup_function():
    """Clear registry before each test."""
    _REGISTRY.clear()


def test_aliases_from_class_name():
    """Test aliases_from_class_name function."""

    # Test normal CamelCase
    aliases = aliases_from_class_name("MyTestClass")
    expected = {"MyTestClass", "my-test-class", "my_test_class", "mytestclass"}
    assert aliases == expected

    # Test single word
    aliases = aliases_from_class_name("Test")
    expected = {"Test", "test"}
    assert aliases == expected
    assert len(aliases) == 2

    # Test already lowercase
    aliases = aliases_from_class_name("test")
    expected = {"test"}
    assert aliases == expected

    # Test with numbers
    aliases = aliases_from_class_name("Test123Class")
    expected = {"Test123Class", "test123-class", "test123_class", "test123class"}
    assert aliases == expected

    # Test empty string
    aliases = aliases_from_class_name("")
    expected = {""}
    assert aliases == expected

    # Test single character
    aliases = aliases_from_class_name("A")
    expected = {"A", "a"}
    assert aliases == expected

    # Test complex case
    aliases = aliases_from_class_name("HTTPSConnectionManager")
    expected = {"HTTPSConnectionManager", "h-t-t-p-s-connection-manager", "h_t_t_p_s_connection_manager",
                "httpsconnectionmanager"}
    assert aliases == expected


def test_register_decorator_with_abstract_rules():
    """Test register decorator functionality with abstract base & concrete subclass constraints."""

    # Concrete subclass (implements abstract foo) -> OK
    @register(MockBase)
    class TestClass(MockBase):
        def foo(self):
            return "ok"

    registry = get_registered(MockBase)
    assert {"TestClass", "test-class", "test_class", "testclass"}.issubset(set(registry.keys()))
    assert registry["TestClass"] is TestClass

    # Registration with custom aliases
    @register(MockBase, aliases=["custom", "alias"])  # base is abstract, subclass is concrete
    class AnotherTestClass(MockBase):
        def foo(self):
            return "ok"

    registry = get_registered(MockBase)
    assert {"custom", "alias", "another-test-class", "another_test_class", "anothertestclass"}.issubset(set(registry.keys()))
    assert registry["AnotherTestClass"] is AnotherTestClass
    assert registry["custom"] is AnotherTestClass
    assert registry["alias"] is AnotherTestClass

    # Error: subclass is abstract -> should fail
    with pytest.raises(TypeError, match=r"AbstractChild cannot be abstract when registered\."):
        @register(MockBase)
        class AbstractChild(MockBase):
            @abstractmethod
            def foo(self):
                pass  # still abstract

    # Error: base is NOT abstract -> should fail
    class NonAbstractBase:
        pass

    with pytest.raises(TypeError, match=r"NonAbstractBase must be an abstract class for registration\."):
        @register(NonAbstractBase)
        class SomeClass(NonAbstractBase):
            pass

    # Error: inheritance check still enforced
    with pytest.raises(TypeError, match=r"NotMockBase must inherit MockBase"):
        @register(MockBase)
        class NotMockBase:
            pass


def test_get_registered():
    """Test get_registered function."""

    # Test empty registry
    assert get_registered(MockBase) == {}

    # Test after registration
    @register(MockBase)
    class RegisteredClass(MockBase):
        def foo(self):
            return 0

    registry = get_registered(MockBase)
    # 4 aliases for RegisteredClass
    assert len(registry) == 4
    assert registry["RegisteredClass"] is RegisteredClass

    # Test unregistered base class
    class UnregisteredBase(ABC):
        @abstractmethod
        def baz(self):
            pass

    assert get_registered(UnregisteredBase) == {}


def test_derived_class_factory():
    """Test derived_class_factory function."""

    @register(MockBase)
    class FactoryTestClass(MockBase):
        def __init__(self, value, keyword=None):
            self.value = value
            self.keyword = keyword
        def foo(self):
            return self.value

    # Test creation with CamelCase name
    instance = derived_class_factory("FactoryTestClass", MockBase, 42, keyword="test")
    assert isinstance(instance, FactoryTestClass)
    assert instance.value == 42
    assert instance.keyword == "test"

    # Test creation with kebab-case name
    instance = derived_class_factory("factory-test-class", MockBase, 100)
    assert isinstance(instance, FactoryTestClass)
    assert instance.value == 100
    assert instance.keyword is None

    # Test creation with snake_case name
    instance = derived_class_factory("factory_test_class", MockBase, "string_value")
    assert isinstance(instance, FactoryTestClass)
    assert instance.value == "string_value"

    # Test creation with lowercase name
    instance = derived_class_factory("factorytestclass", MockBase, [1, 2, 3])
    assert isinstance(instance, FactoryTestClass)
    assert instance.value == [1, 2, 3]

    # Test with custom alias
    @register(MockBase, aliases=["special"])  # concrete
    class AliasedClass(MockBase):
        def __init__(self, data):
            self.data = data
        def foo(self):
            return self.data

    instance = derived_class_factory("special", MockBase, "custom_data")
    assert isinstance(instance, AliasedClass)
    assert instance.data == "custom_data"

    # Test unregistered class
    with pytest.raises(NotImplementedError, match=r"UnknownClass is not implemented\."):
        derived_class_factory("UnknownClass", MockBase)

    # Test constructor errors are propagated
    @register(MockBase)
    class ErrorClass(MockBase):
        def __init__(self):
            raise ValueError("Constructor error")
        def foo(self):
            return 0

    with pytest.raises(ValueError, match="Constructor error"):
        derived_class_factory("ErrorClass", MockBase)


def test_load_plugins_import_errors_debug():
    """Debug version to see what's happening."""

    mock_package = MagicMock(spec=ModuleType)
    mock_package.__name__ = "test_package"
    mock_package.__path__ = ["/fake/path"]

    with (
        patch('pkgutil.walk_packages') as mock_walk,
        patch('importlib.import_module') as mock_import,
        patch('builtins.print') as mock_print
    ):
        mock_mod1 = MagicMock()
        mock_mod1.name = "test_package.good_module"
        mock_mod2 = MagicMock()
        mock_mod2.name = "test_package.bad_module"

        mock_walk.return_value = iter([mock_mod1, mock_mod2])

        good_module = MagicMock()

        def import_side_effect(name):
            print(f"DEBUG: Trying to import {name}")
            if name == "test_package.good_module":
                return good_module
            elif name == "test_package.bad_module":
                raise ImportError("Failed to import")
            else:
                print(f"DEBUG: Unexpected import call for {name}")
                return MagicMock()

        mock_import.side_effect = import_side_effect

        result = load_plugins(mock_package, strict=False)

        print(f"DEBUG: Result = {result}")
        print(f"DEBUG: Print call count = {mock_print.call_count}")
        print(f"DEBUG: Print calls = {mock_print.call_args_list}")
        print(f"DEBUG: Import call count = {mock_import.call_count}")
        print(f"DEBUG: Import calls = {mock_import.call_args_list}")


def test_load_plugins_with_string_package():
    """Test load_plugins with string package name."""

    # Test loading a real module
    with (
        patch('pkgutil.walk_packages') as mock_walk,
        patch('importlib.import_module') as mock_import
    ):
        # Mock the package
        mock_package = MagicMock()
        mock_package.__name__ = "test_package"
        mock_package.__path__ = ["/fake/path"]
        mock_import.return_value = mock_package

        # Mock walk_packages to return some modules
        mock_modinfo1 = MagicMock()
        mock_modinfo1.name = "test_package.module1"
        mock_modinfo2 = MagicMock()
        mock_modinfo2.name = "test_package.module2"
        mock_walk.return_value = [mock_modinfo1, mock_modinfo2]

        # Mock importing the modules
        mock_mod1 = MagicMock()
        mock_mod2 = MagicMock()
        mock_import.side_effect = [mock_package, mock_mod1, mock_mod2]

        result = load_plugins("test_package")

        assert result == ["test_package.module1", "test_package.module2"]
        assert mock_import.call_count == 3  # package + 2 modules


def test_load_plugins_with_module_object():
    """Test load_plugins with ModuleType object."""

    mock_package = MagicMock(spec=ModuleType)
    mock_package.__name__ = "test_package"
    mock_package.__path__ = ["/fake/path"]

    with (
        patch('pkgutil.walk_packages') as mock_walk,
        patch('importlib.import_module') as mock_import
    ):
        mock_modinfo = MagicMock()
        mock_modinfo.name = "test_package.module1"
        mock_walk.return_value = [mock_modinfo]

        mock_mod = MagicMock()
        mock_import.return_value = mock_mod

        result = load_plugins(mock_package)

        assert result == ["test_package.module1"]


def test_load_plugins_no_path():
    """Test load_plugins with module that has no __path__."""

    mock_module = MagicMock()
    mock_module.__name__ = "single_module"
    try:
        del mock_module.__path__  # Remove __path__ attribute if present
    except AttributeError:
        pass

    result = load_plugins(mock_module)
    assert result == ["single_module"]


def test_load_plugins_with_filters():
    """Test load_plugins with include and exclude filters."""

    mock_package = MagicMock(spec=ModuleType)
    mock_package.__name__ = "test_package"
    mock_package.__path__ = ["/fake/path"]

    with (
        patch('pkgutil.walk_packages') as mock_walk,
        patch('importlib.import_module') as mock_import
    ):
        # Create multiple modules
        modules = []
        for i in range(5):
            mock_mod = MagicMock()
            mock_mod.name = f"test_package.module{i}"
            modules.append(mock_mod)
        mock_walk.return_value = modules

        mock_imported = MagicMock()
        mock_import.return_value = mock_imported

        # Test include filter
        def include_even(name):
            return "module" in name and int(name[-1]) % 2 == 0

        result = load_plugins(mock_package, include=include_even)
        expected = ["test_package.module0", "test_package.module2", "test_package.module4"]
        assert result == expected

        # Reset mock
        mock_import.reset_mock()

        # Test exclude filter
        def exclude_odd(name):
            return "module" in name and int(name[-1]) % 2 == 1

        result = load_plugins(mock_package, exclude=exclude_odd)
        expected = ["test_package.module0", "test_package.module2", "test_package.module4"]
        assert result == expected


def test_load_plugins_import_errors():
    """Test load_plugins error handling."""
    import logging

    mock_package = MagicMock(spec=ModuleType)
    mock_package.__name__ = "test_package"
    mock_package.__path__ = ["/fake/path"]

    with (
        patch('pkgutil.walk_packages') as mock_walk,
        patch('importlib.import_module') as mock_import,
    ):
        # Create a mock logger
        mock_logger = MagicMock()

        mock_mod1 = MagicMock()
        mock_mod1.name = "test_package.good_module"
        mock_mod2 = MagicMock()
        mock_mod2.name = "test_package.bad_module"

        mock_walk.return_value = iter([mock_mod1, mock_mod2])

        good_module = MagicMock()
        mock_import.side_effect = [good_module, ImportError("Failed to import")]

        # Test non-strict mode with custom logger
        result = load_plugins(mock_package, strict=False, logger=mock_logger)

        assert result == ["test_package.good_module"]
        mock_logger.error.assert_called_once_with(
            "[load_plugins] Failed to import test_package.bad_module: Failed to import")

        # Reset mocks for strict mode test
        mock_import.reset_mock()
        mock_walk.reset_mock()
        mock_walk.return_value = iter([mock_mod1, mock_mod2])
        mock_import.side_effect = [good_module, ImportError("Failed to import")]

        # Test strict mode - this should raise an exception
        with pytest.raises(RuntimeError, match="Failed to import test_package.bad_module"):
            load_plugins(mock_package, strict=True, logger=mock_logger)


def test_load_plugins_hook_errors():
    """Test load_plugins when hooks raise errors."""
    import logging

    mock_package = MagicMock(spec=ModuleType)
    mock_package.__name__ = "test_package"
    mock_package.__path__ = ["/fake/path"]

    def failing_before_hook(name):
        raise ValueError("Before hook failed")

    def failing_after_hook(name, module):
        raise ValueError("After hook failed")

    with (
        patch('pkgutil.walk_packages') as mock_walk,
        patch('importlib.import_module') as mock_import,
    ):
        mock_logger = MagicMock()
        mock_modinfo = MagicMock()
        mock_modinfo.name = "test_package.module1"
        mock_walk.return_value = [mock_modinfo]

        mock_mod = MagicMock()
        mock_import.return_value = mock_mod

        # Test before_hook error in non-strict mode
        result = load_plugins(
            mock_package,
            before_import=failing_before_hook,
            strict=False,
            logger=mock_logger
        )

        assert result == []
        mock_logger.error.assert_called_once()
        assert "Before hook failed" in str(mock_logger.error.call_args)

        # Reset mocks
        mock_logger.reset_mock()
        mock_import.reset_mock()
        mock_import.return_value = mock_mod

        # Test after_hook error in non-strict mode
        result = load_plugins(
            mock_package,
            after_import=failing_after_hook,
            strict=False,
            logger=mock_logger
        )

        assert result == ["test_package.module1"]  # Module imported successfully
        mock_logger.error.assert_called_once()
        assert "After hook failed" in str(mock_logger.error.call_args)


def test_registry_isolation():
    """Test that different base classes have isolated registries."""

    class BaseA(ABC):
        @abstractmethod
        def a(self): ...
    class BaseB(ABC):
        @abstractmethod
        def b(self): ...

    @register(BaseA)
    class ClassA(BaseA):
        def a(self): return 1

    @register(BaseB)
    class ClassB(BaseB):
        def b(self): return 2

    # Multiple registration allowed for the same class,
    # as long as it is non-abstract subclass of
    # different abstract base classes.
    @register(BaseA)
    @register(BaseB)
    class ClassAB(BaseA, BaseB):
        def a(self): return 1
        def b(self): return 2

    registry_a = get_registered(BaseA)
    registry_b = get_registered(BaseB)

    assert "ClassA" in registry_a
    assert "ClassA" not in registry_b
    assert "ClassB" in registry_b
    assert "ClassB" not in registry_a
    assert "ClassAB" in registry_a
    assert "ClassAB" in registry_b


def test_register_overwrite():
    """Test that registering same alias overwrites previous registration."""

    @register(MockBase, aliases=["shared"])
    class FirstClass(MockBase):
        def foo(self): return 1

    @register(MockBase, aliases=["shared"], overwrite=True)
    class SecondClass(MockBase):
        def foo(self): return 2

    registry = get_registered(MockBase,)
    assert registry["shared"] is SecondClass  # Should be overwritten
    assert registry["first-class"] is FirstClass  # Original, should not be overwritten.


def test_register_no_overwrite():
    """Test that registering same alias overwrites previous registration."""

    @register(MockBase, aliases=["shared"])
    class FirstClass(MockBase):
        def foo(self): return 1

    with pytest.raises(ValueError, match=r"Alias 'shared' already registered"):
        @register(MockBase, aliases=["shared"])
        class SecondClass(MockBase):
            def foo(self): return 2


def test_derived_class_factory_edge_cases():
    """Test edge cases for derived_class_factory."""

    # Test with empty base registry
    class EmptyBase(ABC):
        @abstractmethod
        def z(self): ...

    with pytest.raises(NotImplementedError, match=r"AnyClass is not implemented\."):
        derived_class_factory("AnyClass", EmptyBase)

    # Test class that takes no arguments
    @register(MockBase)
    class NoArgsClass(MockBase):
        def __init__(self):
            self.initialized = True
        def foo(self): return 0

    instance = derived_class_factory("NoArgsClass", MockBase)
    assert instance.initialized is True

    # Test with *args and **kwargs
    @register(MockBase)
    class FlexibleClass(MockBase):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
        def foo(self): return 0

    instance = derived_class_factory(
        "FlexibleClass", MockBase,
        1, 2, 3,
        key1="value1", key2="value2"
    )
    assert instance.args == (1, 2, 3)
    assert instance.kwargs == {"key1": "value1", "key2": "value2"}

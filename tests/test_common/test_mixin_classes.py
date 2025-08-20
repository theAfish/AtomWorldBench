"""Test functions of mixin classes."""

from AtomWorldBench.common.mixin_classes import MultiModeInitMixin

import pytest


class TestClass(MultiModeInitMixin):
    """Test class to demonstrate MultiModeInitMixin functionality."""
    kwargs_formatting_functions = {
        # Always post-process attribute_a.
        "attribute_a": lambda x: x - 1,
    }
    mode_definitions = {
        "_excluded": ["attribute_a"],
        # Fixed modes.
        "abcd": {
            "attribute_b": (
                lambda b: b > 0,
                "b > 0",
            ),
            "attribute_c": (
                lambda c: c > 0,
                "c > 0",
            ),
            "attribute_d": (
                lambda d: d > 0,
                "d > 0",
            ),
            "attribute_e": (
                lambda e: e > 0,
                "e > 0",
            ),
        },
        # Combination modes.
        "_combinations": [
            {
                "name_template": "{b_val}_and_{c_val}",
                "b_val": {
                    "large_b": {
                        "attribute_b": (
                            lambda b: b > 10,
                            "b > 10",
                        )
                    },
                    "small_b": {
                        "attribute_b": (
                            lambda b: b <= 10,
                            "b <= 10",
                        )
                    },
                },
                "c_val": {
                    "large_c": {
                        "attribute_c": (
                            lambda c: c > 5,
                            "c > 10",
                        )
                    },
                    "small_c": {
                        "attribute_c": (
                            lambda c: c <= 5,
                            "c <= 10",
                        )
                    },
                },
            },
            {
                # No name_template, will just concatenate names.
                "d_val": {
                    "large_d": {
                        "attribute_d": (
                            lambda d: d > 10,
                            "d > 10",
                        )
                    },
                    "small_d": {
                        "attribute_d": (
                            lambda d: d <= 10,
                            "d <= 10",
                        )
                    },
                },
                "e_val": {
                    "large_e": {
                        "attribute_e": (
                            lambda e: e > 5,
                            "e > 10",
                        )
                    },
                    "small_e": {
                        "attribute_e": (
                            lambda e: e <= 5,
                            "e <= 10",
                        )
                    },
                },
            },
        ]
    }

    def __init__(
            self,
            attribute_a,
            attribute_b=None,
            attribute_c=None,
            attribute_d=None,
            attribute_e=None,
    ):
        super().__init__(
            attribute_a=attribute_a,
            attribute_b=attribute_b,
            attribute_c=attribute_c,
            attribute_d=attribute_d,
            attribute_e=attribute_e,
        )


def test_multi_mode_init_mixin():
    """Test MultiModeInitMixin functionality."""

    # Test fixed mode initialization.
    test_instance = TestClass(
        attribute_a=5,
        attribute_b=10,
        attribute_c=15,
        attribute_d=20,
        attribute_e=25,
    )
    assert test_instance.attribute_a == 4  # Post-processed by kwargs_formatting_functions
    assert test_instance.attribute_b == 10
    assert test_instance.attribute_c == 15
    assert test_instance.attribute_d == 20
    assert test_instance.attribute_e == 25
    assert test_instance.mode_flag == "abcd"

    # Test combination[0] modes.
    # large, large
    test_instance = TestClass(
        5,     # attribute_a
        attribute_b=15,  # large_b
        attribute_c=10,  # large_c
    )
    assert test_instance.attribute_a == 4
    assert test_instance.attribute_b == 15
    assert test_instance.attribute_c == 10
    assert test_instance.mode_flag == "large_b_and_large_c"
    assert getattr(test_instance, "attribute_d", None) is None
    assert getattr(test_instance, "attribute_e", None) is None

    # small, small
    test_instance = TestClass(
        5,     # attribute_a
        attribute_b=5,  # large_b
        attribute_c=4,  # large_c
    )
    assert test_instance.attribute_a == 4
    assert test_instance.attribute_b == 5
    assert test_instance.attribute_c == 4
    assert test_instance.mode_flag == "small_b_and_small_c"
    assert getattr(test_instance, "attribute_d", None) is None
    assert getattr(test_instance, "attribute_e", None) is None

    # large, small
    test_instance = TestClass(
        5,     # attribute_a
        attribute_b=15,  # large_b
        attribute_c=4,  # small_c
    )
    assert test_instance.attribute_a == 4
    assert test_instance.attribute_b == 15
    assert test_instance.attribute_c == 4
    assert test_instance.mode_flag == "large_b_and_small_c"
    assert getattr(test_instance, "attribute_d", None) is None
    assert getattr(test_instance, "attribute_e", None) is None

    # small, large
    test_instance = TestClass(
        5,     # attribute_a
        attribute_b=5,  # small_b
        attribute_c=10,  # large_c
    )
    assert test_instance.attribute_a == 4
    assert test_instance.attribute_b == 5
    assert test_instance.attribute_c == 10
    assert test_instance.mode_flag == "small_b_and_large_c"
    assert getattr(test_instance, "attribute_d", None) is None
    assert getattr(test_instance, "attribute_e", None) is None

    # Test combination[1] modes.
    # large, large
    test_instance = TestClass(
        5,     # attribute_a
        attribute_d=15,  # large_d
        attribute_e=10,  # large_e
    )
    assert test_instance.attribute_a == 4
    assert test_instance.attribute_d == 15
    assert test_instance.attribute_e == 10
    assert test_instance.mode_flag == "large_d_large_e"  # No name_template, concatenated names.
    assert getattr(test_instance, "attribute_b", None) is None
    assert getattr(test_instance, "attribute_c", None) is None

    # small, small
    test_instance = TestClass(
        5,     # attribute_a
        attribute_d=5,  # small_d
        attribute_e=4,  # small_e
    )
    assert test_instance.attribute_a == 4
    assert test_instance.attribute_d == 5
    assert test_instance.attribute_e == 4
    assert test_instance.mode_flag == "small_d_small_e"  # No name_template, concatenated names.
    assert getattr(test_instance, "attribute_b", None) is None
    assert getattr(test_instance, "attribute_c", None) is None

    # large, small
    test_instance = TestClass(
        5,     # attribute_a
        attribute_d=15,  # large_d
        attribute_e=4,  # small_e
    )
    assert test_instance.attribute_a == 4
    assert test_instance.attribute_d == 15
    assert test_instance.attribute_e == 4
    assert test_instance.mode_flag == "large_d_small_e"  # No name_template, concatenated names.
    assert getattr(test_instance, "attribute_b", None) is None
    assert getattr(test_instance, "attribute_c", None) is None

    # small, large
    test_instance = TestClass(
        5,     # attribute_a
        attribute_d=5,  # small_d
        attribute_e=10,  # large_e
    )
    assert test_instance.attribute_a == 4
    assert test_instance.attribute_d == 5
    assert test_instance.attribute_e == 10
    assert test_instance.mode_flag == "small_d_large_e"  # No name_template,
    assert getattr(test_instance, "attribute_b", None) is None
    assert getattr(test_instance, "attribute_c", None) is None

    # Error case 1: attribute a not provided.
    with pytest.raises(TypeError):  # attribute_a is required.
        _ = TestClass(
            attribute_b=10,
            attribute_c=15,
            attribute_d=20,
            attribute_e=25,
        )

    with pytest.raises(TypeError): # None cannot be subtracted by 1.
        _ = TestClass(
            None,
            attribute_b=10,
            attribute_c=15,
            attribute_d=20,
            attribute_e=25,
        )

    # Error case 2: no valid case.
    with pytest.raises(ValueError):
        _ = TestClass(
            5,
            attribute_b=-10,  # Invalid value for b.
            attribute_c=15,
            attribute_d=20,
            attribute_e=25,
        )

    with pytest.raises(ValueError):
        _ = TestClass(
            5,
            attribute_b=10,
            attribute_c=15,
            attribute_d=20,
        ) # Missing e, or extra d.


def test_init_subclass_validation_errors():
    """Test various validation errors in __init_subclass__."""

    # Test missing mode_definitions
    with pytest.raises(TypeError, match="mode_definitions must be a mapping"):
        class BadClass1(MultiModeInitMixin):
            mode_definitions = None  # No mode_definitions

    # Test non-mapping mode_definitions
    with pytest.raises(TypeError, match="mode_definitions must be a mapping"):
        class BadClass2(MultiModeInitMixin):
            mode_definitions = "not a mapping"

    # Test missing _excluded key
    with pytest.raises(TypeError, match="must include the '_excluded' key"):
        class BadClass3(MultiModeInitMixin):
            mode_definitions = {"some_mode": {}}

    # Test non-sequence _excluded
    with pytest.raises(TypeError, match="_excluded must be a sequence"):
        class BadClass4(MultiModeInitMixin):
            mode_definitions = {"_excluded": 1} # Not a sequence.

    # Test non-mapping explicit mode
    with pytest.raises(TypeError, match="must be a mapping of param -> Condition"):
        class BadClass5(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "bad_mode": "not a mapping"
            }

    # Test excluded parameter in explicit mode
    with pytest.raises(ValueError, match="is both in mode .* and in _excluded"):
        class BadClass6(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": ["param_a"],
                "mode1": {"param_a": None}
            }

    # Test invalid condition format
    with pytest.raises(TypeError, match="must be None or \\(callable, 'description'\\)"):
        class BadClass7(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "mode1": {"param_a": "invalid condition"}
            }

    # Test invalid condition tuple length
    with pytest.raises(TypeError, match="must be None or \\(callable, 'description'\\)"):
        class BadClass8(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "mode1": {"param_a": (lambda x: True,)}  # Missing description
            }

    # Test non-callable in condition tuple
    with pytest.raises(TypeError, match="must be None or \\(callable, 'description'\\)"):
        class BadClass9(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "mode1": {"param_a": ("not callable", "desc")}
            }

    # Test non-string description in condition tuple
    with pytest.raises(TypeError, match="must be None or \\(callable, 'description'\\)"):
        class BadClass10(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "mode1": {"param_a": (lambda x: True, 123)}  # Non-string description
            }


def test_combinations_validation_errors():
    """Test validation errors in _combinations."""

    # Test non-sequence _combinations
    with pytest.raises(TypeError, match="_combinations must be a sequence"):
        class BadCombo1(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "_combinations": 1 # Not a sequence
            }

    # Test non-mapping block in _combinations
    with pytest.raises(TypeError, match="_combinations\\[0\\] must be a mapping"):
        class BadCombo2(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "_combinations": ["not a mapping"]
            }

    # Test non-string name_template
    with pytest.raises(TypeError, match="name_template must be a str"):
        class BadCombo3(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "_combinations": [{
                    "name_template": 123,
                    "dim1": {"opt1": {"param1": None}}
                }]
            }

    # Test empty dimensions block
    with pytest.raises(ValueError, match="must define at least one dimension"):
        class BadCombo4(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "_combinations": [{"name_template": "test"}]
            }

    # Test non-mapping dimension options
    with pytest.raises(TypeError, match="must be a non-empty mapping"):
        class BadCombo5(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "_combinations": [{
                    "dim1": "not a mapping"
                }]
            }

    # Test empty dimension options
    with pytest.raises(TypeError, match="must be a non-empty mapping"):
        class BadCombo6(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "_combinations": [{
                    "dim1": {}
                }]
            }

    # Test non-mapping option rule
    with pytest.raises(TypeError, match="must be a mapping of param -> Condition"):
        class BadCombo7(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "_combinations": [{
                    "dim1": {"opt1": "not a mapping"}
                }]
            }

    # Test excluded parameter in combinations
    with pytest.raises(ValueError, match="is both in _excluded and used in _combinations"):
        class BadCombo8(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": ["param1"],
                "_combinations": [{
                    "dim1": {"opt1": {"param1": None}}
                }]
            }

    # Test invalid condition in combinations
    with pytest.raises(TypeError, match="must be None or \\(callable, 'description'\\)"):
        class BadCombo9(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "_combinations": [{
                    "dim1": {"opt1": {"param1": "invalid"}}
                }]
            }

    # Test parameter collision in combinations
    with pytest.raises(ValueError, match="appears in multiple parts"):
        class BadCombo10(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "_combinations": [{
                    "dim1": {"opt1": {"param1": None}},
                    "dim2": {"opt2": {"param1": None}}  # Same param in different dimensions
                }]
            }

    # Test duplicate mode name from combinations
    with pytest.raises(ValueError, match="duplicate mode name"):
        class BadCombo11(MultiModeInitMixin):
            mode_definitions = {
                "_excluded": [],
                "opt1_opt2": {"existing": None},  # Explicit mode
                "_combinations": [{
                    "dim1": {"opt1": {"param1": None}},
                    "dim2": {"opt2": {"param2": None}}  # Will generate "opt1_opt2"
                }]
            }


def test_formatter_errors():
    """Test formatter-related errors."""

    class TestFormatterClass(MultiModeInitMixin):
        mode_definitions = {
            "_excluded": [],
            "mode1": {"param1": None}
        }
        kwargs_formatting_functions = {
            "param1": "not a callable"
        }

    with pytest.raises(TypeError, match="must be callable"):
        TestFormatterClass(param1="value")


def test_mode_detection_edge_cases():
    """Test edge cases in mode detection."""

    # Test condition that raises an exception
    class BadConditionClass(MultiModeInitMixin):
        mode_definitions = {
            "_excluded": [],
            "mode1": {"param1": (lambda x: x.invalid_method(), "will fail")}
        }

    with pytest.raises(ValueError, match="condition.*raised"):
        BadConditionClass(param1="test")

    # Test ambiguous modes (this shouldn't happen with proper definitions, but test defensive code)
    class AmbiguousClass(MultiModeInitMixin):
        mode_definitions = {
            "_excluded": ["shared"],
            "mode1": {},  # Empty mode
            "mode2": {}   # Another empty mode
        }

    with pytest.raises(ValueError, match="Ambiguous modes"):
        AmbiguousClass(shared="ignored")


def test_mode_flag_immutability():
    """Test that mode_flag cannot be changed after initialization."""

    instance = TestClass(
        attribute_a=5,
        attribute_b=10,
        attribute_c=15,
        attribute_d=20,
        attribute_e=25,
    )

    # Test that we can't set _mode_flag directly
    with pytest.raises(AttributeError, match="mode_flag is immutable"):
        instance._mode_flag = "new_mode"


def test_complex_combinations():
    """Test more complex combination scenarios."""

    class ComplexComboClass(MultiModeInitMixin):
        mode_definitions = {
            "_excluded": ["metadata"],
            "_combinations": [
                {
                    "name_template": "{type}_{size}",
                    "type": {
                        "circle": {"radius": (lambda r: r > 0, "r > 0")},
                        "square": {"side": (lambda s: s > 0, "s > 0")}
                    },
                    "size": {
                        "small": {"max_area": (lambda a: a <= 100, "a <= 100")},
                        "large": {"max_area": (lambda a: a > 100, "a > 100")}
                    }
                }
            ]
        }
        def __init__(
                self,
                radius=None,
                side=None,
                max_area=None,
        ):
            super().__init__(
                radius=radius,
                side=side,
                max_area=max_area,
            )

    # Test successful combinations
    instance = ComplexComboClass(radius=5, max_area=50)
    assert instance.mode_flag == "circle_small"

    instance = ComplexComboClass(side=15, max_area=200)
    assert instance.mode_flag == "square_large"

    # Test validation failures
    with pytest.raises(ValueError, match="No mode detected"):
        ComplexComboClass(radius=-1, max_area=200)  # circle with negative radius - impossible
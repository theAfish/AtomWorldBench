import pytest
from utils.extract_data import extract_from_string


def test_extract_from_string_valid():
    s = "some text <cif>\ndata_line1\ndata_line2\n</cif> trailing"
    extracted = extract_from_string(s, format="cif")
    assert extracted == "data_line1\ndata_line2"

def test_extract_from_string_multiple_tags():
    s = "<cif>first</cif> middle <cif>second</cif>"
    extracted = extract_from_string(s, format="cif")
    assert extracted == "second"


def test_extract_from_string_missing_tags():
    s = "no tags here"
    assert extract_from_string(s, format="cif") is None


def test_extract_from_string_unsupported_format():
    with pytest.raises(ValueError):
        extract_from_string("<xml></xml>", format="xml")

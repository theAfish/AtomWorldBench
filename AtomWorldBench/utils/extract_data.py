

def extract_from_string(data: str, format: str = "cif") -> str:
    """
    Extracts the content from a string based on the specified format.
    
    Args:
        data (str): The input string containing the data.
        format (str): The format to extract from the string (default is "cif").
        
    Returns:
        str: The extracted content in the specified format.
    """
    if format == "cif":
        start_tag = "<cif>"
        end_tag = "</cif>"
    else:
        raise ValueError(f"Unsupported format: {format}")

    start_index = data.rfind(start_tag)
    end_index = data.rfind(end_tag, start_index)

    if start_index == -1 or end_index == -1:
        return None

    return data[start_index + len(start_tag):end_index].strip()
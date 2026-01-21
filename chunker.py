import ast
from langchain_core.documents import Document

def chunk_code(file_path: str, code: str):
    """
    Chunk Python code by functions and classes using AST.
    Falls back to whole-file chunk if parsing fails.
    """
    documents = []

    try:
        tree = ast.parse(code)
    except Exception:
        # Fallback: whole file as one chunk
        if len(code.strip()) > 100:
            documents.append(
                Document(
                    page_content=code,
                    metadata={
                        "file": file_path,
                        "type": "file"
                    }
                )
            )
        return documents

    for node in ast.walk(tree):
        # -------- FUNCTIONS --------
        if isinstance(node, ast.FunctionDef):
            source = ast.get_source_segment(code, node)
            if source:
                documents.append(
                    Document(
                        page_content=source,
                        metadata={
                            "file": file_path,
                            "type": "function",
                            "name": node.name,
                            "line_start": node.lineno
                        }
                    )
                )

        # -------- CLASSES --------
        elif isinstance(node, ast.ClassDef):
            source = ast.get_source_segment(code, node)
            if source:
                documents.append(
                    Document(
                        page_content=source,
                        metadata={
                            "file": file_path,
                            "type": "class",
                            "name": node.name,
                            "line_start": node.lineno
                        }
                    )
                )

    # If no functions/classes found, keep whole file
    if not documents and len(code.strip()) > 100:
        documents.append(
            Document(
                page_content=code,
                metadata={
                    "file": file_path,
                    "type": "file"
                }
            )
        )

    return documents

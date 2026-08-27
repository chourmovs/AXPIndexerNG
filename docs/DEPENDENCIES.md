# Dependencies

Direct runtime roots are FastEmbed 0.8.0, sqlite-vec 0.1.9, PyMuPDF, python-docx, and python-pptx. FastEmbed supplies the CPU ONNX inference path; its heavyweight transitives include ONNX Runtime, NumPy, and tokenizers. There is one ONNX Runtime implementation and no GPU/DirectML package.

Native components visible to IT/SOC are the standard WinPython `python.exe`/Python DLL, ONNX Runtime, sqlite-vec's SQLite extension, and PyMuPDF native modules. The application intentionally excludes LanceDB, PyArrow, pandas, scientific-distribution extras, web frameworks, and Node tooling. The embedding model cache is separately provisioned and excluded from the runtime ZIP.

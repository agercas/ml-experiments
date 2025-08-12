# Agent system for GAIA benchmark

# Architecture

```python
gaia_system = create_gaia_system()
gaia_system.visualize_system()
```


```
CodeAgent | Qwen/Qwen2.5-Coder-32B-Instruct
├── ✅ Authorized imports: []
├── 🛠️ Tools:
│   ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│   ┃ Name         ┃ Description                                   ┃ Arguments                                    ┃
│   ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   │ final_answer │ Provides a final answer to the given problem. │ answer (`any`): The final answer to the      │
│   │              │                                               │ problem                                      │
│   └──────────────┴───────────────────────────────────────────────┴──────────────────────────────────────────────┘
└── 🤖 Managed agents:
    ├── search_agent | CodeAgent | Qwen/Qwen2.5-Coder-32B-Instruct
    │   ├── ✅ Authorized imports: []
    │   ├── 📝 Description: Retrieves factual information and background data from various sources including 
    │   │   Wikipedia, web search, and academic papers
    │   └── 🛠️ Tools:
    │       ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    │       ┃ Name                  ┃ Description                          ┃ Arguments                            ┃
    │       ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       │ wikipedia_search      │ Search Wikipedia for information on  │ query (`string`): The search term or │
    │       │                       │ a specific topic.                    │ topic                                │
    │       │                       │                                      │ language (`string`): Wikipedia       │
    │       │                       │                                      │ language code (default: "en")        │
    │       │                       │                                      │ sentences (`integer`): Number of     │
    │       │                       │                                      │ sentences to return from summary     │
    │       │                       │                                      │ (default: 3)                         │
    │       │ web_search_duckduckgo │ Search the web using DuckDuckGo      │ query (`string`): Search query       │
    │       │                       │ search engine.                       │ string                               │
    │       │                       │                                      │ max_results (`integer`): Maximum     │
    │       │                       │                                      │ number of results to return          │
    │       │                       │                                      │ (default: 5)                         │
    │       │ fetch_webpage_content │ Fetch and extract text content from  │ url (`string`): The URL to fetch     │
    │       │                       │ a webpage.                           │ max_length (`integer`): Maximum      │
    │       │                       │                                      │ length of content to return          │
    │       │                       │                                      │ (default: 3000)                      │
    │       │ arxiv_search          │ Search arXiv papers.                 │ query (`string`): Search query or    │
    │       │                       │                                      │ paper ID (e.g., "1605.08386")        │
    │       │ wikipedia_search_tool │ Search Wikipedia using LangChain's   │ query (`string`): Search query       │
    │       │                       │ WikipediaQueryRun.                   │                                      │
    │       │ duckduckgo_search     │ Search using DuckDuckGo.             │ query (`string`): Search query       │
    │       │ final_answer          │ Provides a final answer to the given │ answer (`any`): The final answer to  │
    │       │                       │ problem.                             │ the problem                          │
    │       └───────────────────────┴──────────────────────────────────────┴──────────────────────────────────────┘
    ├── document_agent | CodeAgent | Qwen/Qwen2.5-Coder-32B-Instruct
    │   ├── ✅ Authorized imports: []
    │   ├── 📝 Description: Loads and processes structured and unstructured documents including CSV, Excel, text 
    │   │   files, and audio transcriptions
    │   └── 🛠️ Tools:
    │       ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    │       ┃ Name                  ┃ Description                          ┃ Arguments                            ┃
    │       ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       │ load_csv_file         │ Load and analyze a CSV file.         │ filepath (`string`): Path to the CSV │
    │       │                       │                                      │ file                                 │
    │       │                       │                                      │ max_rows (`integer`): Maximum number │
    │       │                       │                                      │ of rows to display (default: 100)    │
    │       │                       │                                      │ max_columns (`integer`): Maximum     │
    │       │                       │                                      │ number of columns to display         │
    │       │                       │                                      │ (default: 20)                        │
    │       │                       │                                      │ get_all_rows (`boolean`): If True,   │
    │       │                       │                                      │ return all rows regardless of        │
    │       │                       │                                      │ max_rows (default: False)            │
    │       │ load_excel_file       │ Load and analyze an Excel file.      │ filepath (`string`): Path to the     │
    │       │                       │                                      │ Excel file                           │
    │       │                       │                                      │ sheet_name (`string`): Specific      │
    │       │                       │                                      │ sheet to load (default: None for     │
    │       │                       │                                      │ first sheet)                         │
    │       │                       │                                      │ max_rows (`integer`): Maximum number │
    │       │                       │                                      │ of rows to display (default: 100)    │
    │       │                       │                                      │ max_columns (`integer`): Maximum     │
    │       │                       │                                      │ number of columns to display         │
    │       │                       │                                      │ (default: 20)                        │
    │       │                       │                                      │ get_all_rows (`boolean`): If True,   │
    │       │                       │                                      │ return all rows regardless of        │
    │       │                       │                                      │ max_rows (default: False)            │
    │       │ read_text_file        │ Read content from a text file.       │ filepath (`string`): Path to the     │
    │       │                       │                                      │ text file                            │
    │       │                       │                                      │ max_length (`integer`): Maximum      │
    │       │                       │                                      │ length of content to return          │
    │       │                       │                                      │ (default: 2000)                      │
    │       │                       │                                      │ encoding (`string`): File encoding   │
    │       │                       │                                      │ (default: "utf-8")                   │
    │       │ transcribe_audio_file │ Transcribe audio file to text using  │ filepath (`string`): Path to the     │
    │       │                       │ Whisper.                             │ audio file                           │
    │       │                       │                                      │ model_size (`string`): Whisper model │
    │       │                       │                                      │ size ("tiny", "base", "small",       │
    │       │                       │                                      │ "medium", "large")                   │
    │       │ final_answer          │ Provides a final answer to the given │ answer (`any`): The final answer to  │
    │       │                       │ problem.                             │ the problem                          │
    │       └───────────────────────┴──────────────────────────────────────┴──────────────────────────────────────┘
    ├── vision_agent | CodeAgent | Qwen/Qwen2.5-Coder-32B-Instruct
    │   ├── ✅ Authorized imports: []
    │   ├── 📝 Description: Extracts text and meaning from images using OCR, captioning, and visual question 
    │   │   answering
    │   └── 🛠️ Tools:
    │       ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    │       ┃ Name                  ┃ Description                          ┃ Arguments                            ┃
    │       ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       │ ocr_tool              │ Extract text from images using OCR.  │ image_path (`string`): Path to image │
    │       │                       │                                      │ file                                 │
    │       │ image_captioning_tool │ Generate basic image information     │ image_path (`string`): Path to image │
    │       │                       │ (placeholder for actual captioning). │ file                                 │
    │       │ visual_qa_tool        │ Answer questions about images        │ image_path (`string`): Path to image │
    │       │                       │ (placeholder for actual VQA).        │ file                                 │
    │       │                       │                                      │ question (`string`): Question about  │
    │       │                       │                                      │ the image                            │
    │       │ final_answer          │ Provides a final answer to the given │ answer (`any`): The final answer to  │
    │       │                       │ problem.                             │ the problem                          │
    │       └───────────────────────┴──────────────────────────────────────┴──────────────────────────────────────┘
    ├── reasoning_agent | CodeAgent | Qwen/Qwen2.5-Coder-32B-Instruct
    │   ├── ✅ Authorized imports: []
    │   ├── 📝 Description: Performs symbolic reasoning, logical pattern recognition, and analytical tasks
    │   └── 🛠️ Tools:
    │       ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    │       ┃ Name                        ┃ Description                       ┃ Arguments                         ┃
    │       ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       │ analyze_chess_position      │ Analyze a chess position given in │ fen_notation (`string`): Chess    │
    │       │                             │ FEN notation.                     │ position in FEN (Forsyth-Edwards  │
    │       │                             │                                   │ Notation)                         │
    │       │ analyze_table_commutativity │ Analyze a mathematical operation  │ table_data (`string`): String     │
    │       │                             │ table for commutativity.          │ representation of the operation   │
    │       │                             │                                   │ table                             │
    │       │ count_items_in_list         │ Count items in a delimited list.  │ items_text (`string`): Text       │
    │       │                             │                                   │ containing delimited items        │
    │       │                             │                                   │ separator (`string`): Delimiter   │
    │       │                             │                                   │ to split on (default: ",")        │
    │       │ final_answer                │ Provides a final answer to the    │ answer (`any`): The final answer  │
    │       │                             │ given problem.                    │ to the problem                    │
    │       └─────────────────────────────┴───────────────────────────────────┴───────────────────────────────────┘
    ├── language_agent | CodeAgent | Qwen/Qwen2.5-Coder-32B-Instruct
    │   ├── ✅ Authorized imports: []
    │   ├── 📝 Description: Handles low-level text transformations and string manipulations
    │   └── 🛠️ Tools:
    │       ┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    │       ┃ Name                    ┃ Description                         ┃ Arguments                           ┃
    │       ┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       │ reverse_string          │ Reverse a string character by       │ text (`string`): The string to      │
    │       │                         │ character.                          │ reverse                             │
    │       │ reverse_words_in_string │ Reverse the order of words in a     │ text (`string`): The string with    │
    │       │                         │ string.                             │ words to reverse                    │
    │       │ final_answer            │ Provides a final answer to the      │ answer (`any`): The final answer to │
    │       │                         │ given problem.                      │ the problem                         │
    │       └─────────────────────────┴─────────────────────────────────────┴─────────────────────────────────────┘
    └── coding_agent | CodeAgent | Qwen/Qwen2.5-Coder-32B-Instruct
        ├── ✅ Authorized imports: ['pandas', 'numpy', 'matplotlib', 'json', 're', 'datetime', 'math', 
        │   'statistics', 'itertools']
        ├── 📝 Description: Executes Python code and performs computational logic through code interpretation
        └── 🛠️ Tools:
            ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃ Name         ┃ Description                               ┃ Arguments                                ┃
            ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │ final_answer │ Provides a final answer to the given      │ answer (`any`): The final answer to the  │
            │              │ problem.                                  │ problem                                  │
            └──────────────┴───────────────────────────────────────────┴──────────────────────────────────────────┘
```

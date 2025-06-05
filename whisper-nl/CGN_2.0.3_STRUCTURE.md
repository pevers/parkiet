# CGN 2.0.3 - Corpus Gesproken Nederlands (Spoken Dutch Corpus) - Structure Documentation

## Overview

The **Corpus Gesproken Nederlands (CGN)** version 2.0.3 is a comprehensive spoken Dutch corpus developed through a joint Flemish-Dutch project conducted between 1998 and 2003. This corpus contains approximately **8.9 million words** of speech data with multiple annotation layers.

### Copyright and Licensing
- **Copyright**: Dutch Language Union (Nederlandse Taalunie), March 2004
- **Usage**: 
  - Permitted for scientific research and non-commercial product development
  - Commercial use allowed for speech recognizers and language models
  - Individual contributions must not be recognizable
- **Contact**: taalmaterialen@ivdnt.org

---

## Directory Structure

```
CGN_2.0.3/
├── info.txt                 # Main overview and organization info
├── copyright.txt            # Copyright and licensing information
├── corex/                   # COREX corpus exploitation software (OUTDATED)
├── data/                    # All corpus data (main content)
├── doc_Dutch/              # Dutch documentation
├── doc_English/            # English documentation  
└── tools/                  # Various software utilities
```

---

## 1. Main Data Directory (`/data/`)

The core corpus content is organized into four main subdirectories:

### 1.1 Audio Data (`/data/audio/`)
- **Location**: `/data/audio/wav/` (suggested location for WAV files)
- **Components**: 15 different speech types (comp-a through comp-o)
- **Format**: WAV audio files
- **Note**: Audio files are distributed separately on WAV DVDs

#### Speech Components:
| Component | Description |
|-----------|-------------|
| comp-a | Spontaneous conversations (face-to-face) |
| comp-b | Interviews with teachers of Dutch |
| comp-c | Spontaneous telephone dialogues (via switchboard) |
| comp-d | Spontaneous telephone dialogues (MD with local interface) |
| comp-e | Simulated business negotiations |
| comp-f | Interviews/discussions/debates (broadcast) |
| comp-g | Political discussions/debates/meetings (non-broadcast) |
| comp-h | Lessons recorded in classroom |
| comp-i | Live commentaries, e.g., sports (broadcast) |
| comp-j | News reports/reportages (broadcast) |
| comp-k | News (broadcast) |
| comp-l | Commentaries/columns/reviews (broadcast) |
| comp-m | Ceremonious speeches/sermons |
| comp-n | Lectures/seminars |
| comp-o | Read speech |

### 1.2 Annotations (`/data/annot/`)

Multiple annotation layers available in three formats:

#### Format Types:
- **`/text/`**: Plain text format annotations
- **`/xml/`**: XML format annotations (required for COREX)
- **`/corex/`**: COREX-specific format for corpus exploitation
- **`/dtd/`**: Document Type Definitions for XML validation

#### Annotation Layers:
1. **Orthographic transcriptions** - Basic speech-to-text transcription
2. **Part-of-speech tagging and lemmatisation** - Grammatical analysis
3. **Lexicon link-up** (+ multi-word structures) - Dictionary connections
4. **Automatic word segmentation** (+ automatic phonetic transcription)
5. **Syntactic annotations** - Grammatical structure analysis
6. **Broad phonetic transcriptions** - Phonemic representations
7. **Manually verified word segmentations** - Human-corrected boundaries
8. **Prosodic transcriptions** - Intonation and rhythm patterns

### 1.3 Metadata (`/data/meta/`)

Information about speakers and recordings in multiple formats:

#### Format Types:
- **`/text/`**: TAB-separated text files
  - `/regioncodes/`: Country codes and postal codes
- **`/xls/`**: Excel format files
  - `/regioncodes/`: Country codes and postal codes  
- **`/imdi/`**: IMDI XML format (used by COREX)

### 1.4 Lexicon (`/data/lexicon/`)

CGN lexicon and frequency information:

#### Components:
- **`/text/`**: Text format lexicons
  - `cgnlex_2.0.txt`: Single word lexicon
  - `cgnmlex_2.0.txt`: Multi-word lexicon
- **`/xml/`**: XML format lexicons
  - `cgnlex_2.0.lex`: Single word lexicon
  - `cgnmlex_2.0.lex`: Multi-word lexicon
  - DTD files for validation
- **`/freqlists/`**: Frequency counts for words, tags, lemmas, and phonetic transcriptions

---

## 2. Documentation (`/doc_English/` and `/doc_Dutch/`)

Comprehensive documentation available in both English and Dutch:

### Structure:
- **`start.htm`**: Entry point for documentation
- **`topics/`**: Detailed topic-specific documentation
  - `overview.htm`: Comprehensive corpus overview
  - `annot/`: Annotation documentation
  - `metadata/`: Metadata documentation  
  - `lexicon/`: Lexicon documentation
  - `design/`: Corpus design principles
  - `formats/`: File format specifications
  - `tools/`: Tool documentation
  - `project/`: Project information
  - `other/`: Additional resources
- **`images/`**: Documentation images
- **`styles/`**: CSS stylesheets

---

## 3. COREX Software (`/corex/`)

**⚠️ WARNING: COREX software is OUTDATED and NO LONGER SUPPORTED**

### Purpose:
- Corpus exploitation software for listening to speech files
- View multiple annotations with synchronized audio playback
- Conduct searches and statistical queries
- Navigate through subcorpora based on metadata

### Structure:
- **`scripts/`**: Installation and execution scripts
- **`java/`**: Java runtime for Windows
- **`perl/`**: Perl runtime for Windows  
- **`jars/`**: Java archive files
- **`tiger/`**: Configuration for TIGER syntax search
- **`IMDI-tools/`**: Tools for IMDI metadata format
- **`doc/`**: COREX documentation
- **`ant/`**: Build system components
- **`tools/`**: External utilities
- **`ZIPPED/`**: Compressed data files

---

## 4. Tools (`/tools/`)

Various software utilities organized by function:

### Tool Categories:
- **`/converters/`**: Text file format conversion tools
  - Platform compatibility (Windows/Unix/Mac)
  - Character encoding conversion (ISO/SGML/PRAAT)
  - End-of-line format conversion (CR/LF)

- **`/exploitation/`**: Corpus exploitation software (see COREX section)

- **`/external/`**: External dependencies
  - Perl runtime
  - Tcl/Tk toolkit

- **`/production/`**: CGN production tools
  - Tools used during corpus creation

- **`/viewers/`**: Standalone annotation viewers
  - PRAAT integration
  - PORTRAY text viewer

---

## 5. Corpus Statistics

### Overall Statistics (Version 2.0):
- **Total Words**: 8,940,098
- **Flemish Data**: 3,285,631 words  
- **Dutch Data**: 5,654,644 words

### Data Distribution by Component:
| Component | Total Words | Flemish | Dutch | Description |
|-----------|-------------|---------|-------|-------------|
| a | 2,626,172 | 878,383 | 1,747,789 | Face-to-face conversations |
| b | 565,433 | 315,554 | 249,879 | Teacher interviews |
| c | 1,232,636 | 489,100 | 743,537 | Phone dialogues (switchboard) |
| d | 853,371 | 343,167 | 510,204 | Phone dialogues (MD) |
| e | 136,461 | 0 | 136,461 | Business negotiations |
| f | 790,269 | 250,708 | 539,561 | Broadcast interviews/debates |
| g | 360,328 | 138,819 | 221,509 | Non-broadcast discussions |
| h | 405,409 | 105,436 | 299,973 | Classroom lessons |
| i | 208,399 | 78,022 | 130,377 | Live commentaries |
| j | 186,072 | 95,206 | 90,866 | News reports |
| k | 368,153 | 82,855 | 285,298 | News broadcasts |
| l | 145,553 | 65,386 | 80,167 | Commentaries/reviews |
| m | 18,075 | 12,510 | 5,565 | Ceremonious speeches |
| n | 140,901 | 79,067 | 61,834 | Lectures/seminars |
| o | 903,043 | 351,419 | 551,624 | Read speech |

### Advanced Annotations:
- **Phonetic Transcriptions**: ~675,000 words
- **Syntactic Annotations**: ~668,000 words  
- **Prosodic Annotations**: ~122,000 words

---

## 6. File Formats and Standards

### Audio Format:
- **Format**: WAV files
- **Distribution**: Separate WAV DVDs (CGN_WAV_*)

### Annotation Formats:
- **Text**: Plain text with specific markup
- **XML**: Structured XML with DTD validation
- **COREX**: Proprietary format for COREX software

### Metadata Standards:
- **IMDI**: International Metadata Initiative XML format
- **TAB**: Tab-separated values for tabular data
- **Excel**: Microsoft Excel format (.xls)

---

## 7. Usage Recommendations

### For Research:
1. Start with `/doc_English/start.htm` for documentation
2. Use `/data/annot/xml/` for structured annotation access
3. Reference `/data/meta/` for speaker/recording information
4. Utilize `/data/lexicon/` for vocabulary analysis

### For Development:
1. Use `/tools/converters/` for format compatibility
2. Reference `/data/lexicon/freqlists/` for frequency information
3. Avoid COREX software (deprecated)
4. Use `/data/annot/text/` for simpler text processing

### Important Notes:
- COREX software is no longer maintained or supported
- Some files are compressed (.gz format)
- XML files require corresponding DTD files for validation
- Respect copyright restrictions for individual speaker contributions

---

## 8. Contact Information

- **Website**: https://taalmaterialen.ivdnt.org/
- **Email**: taalmaterialen@ivdnt.org
- **Organization**: Dutch Language Institute (Instituut voor de Nederlandse Taal)

---

*Last Updated: Based on CGN Version 2.0.3 documentation* 
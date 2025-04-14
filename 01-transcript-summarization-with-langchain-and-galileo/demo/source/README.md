# Transcript Summarization Demo

This is a React-based user interface for the `transcript-summarization` example. It allows users to upload text documents (PDF, DOC, TXT formats), extract text from them, and display the API output as a summary.

## Features

- Upload text documents (TXT, PDF, DOC/DOCX)
- Extract text from various file formats
- Submit text for AI summarization
- View both the original text and the summary
- Toggle between detailed and simple views

## Running the Application

1. Install dependencies:

```bash
npm install
```

2. Start the development server:

```bash
npm run dev
```

3. Build for production:

```bash
npm run build
```

4. Preview the production build:

```bash
npm run preview
```

## Interface

The UI provides two viewing modes:
- **Simple View**: Shows only the summary output
- **Detailed View**: Shows both the original text and the generated summary side by side

## API Integration

This UI sends requests to the same API endpoint used by the vanilla-rag example, but with a modified request structure using a `text` key instead of a `question` key in the payload.

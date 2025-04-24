# Image Generation Demo

This is a React-based user interface for the `image-generation` example. It allows users to enter text prompts and customize parameters to generate images using the MLflow model.

## Features

- Input text prompts for image generation
- Customize image parameters (width, height, number of images, inference steps)
- Generate and display AI-created images
- View images in a responsive grid layout
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
- **Simple View**: Shows just the prompt input and generated images
- **Detailed View**: Shows additional parameter controls and detailed information

## API Integration

This UI sends requests to the API endpoint provided by MLFlow. The application uses a RESTful approach to communicate with the backend:

### API Request Format

The application sends POST requests to the `/invocations` endpoint with the following JSON structure:

```json
{
  "prompt": "Your image generation prompt text",
  "height": 512,
  "width": 512,
  "num_images": 1,
  "num_inference_steps": 30
}
```

### API Response Format

The API returns a JSON array of image data or URLs:

```json
[
  "data:image/png;base64,IMAGE_DATA_HERE",
  "data:image/png;base64,IMAGE_DATA_HERE"
]
```

### Successful Demonstration of the User Interface

![Image Generation Demo UI](docs/ui_image_generation.png)

### Implementation Details

- The application takes user input for text prompts and generation parameters
- Parameters are validated before submission to the API
- Generated images are displayed in a responsive grid layout
- The application handles both direct image data and URL responses
- Loading states and error handling are implemented for better user experience

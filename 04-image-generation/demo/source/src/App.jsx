import { useState, useEffect } from 'react';
import { Card, TextBox } from '@veneer/core';
import { IconInfo } from '@veneer/core';
import { Tooltip } from '@veneer/core';
import { Toggle } from '@veneer/core';
import { Button } from '@veneer/core';
import { Slider } from '@veneer/core';
import { ProgressIndicator } from '@veneer/core';
import iconImage from '../icon.ico';
import './App.css';

/**
 * ImageGenerationApp - Main component for the image generation UI
 * @returns {JSX.Element} The rendered component
 */
function App() {
	// State for prompt and parameters
	const [prompt, setPrompt] = useState("");
	const [height, setHeight] = useState(512);
	const [width, setWidth] = useState(512);
	const [numImages, setNumImages] = useState(1);
	const [numInferenceSteps, setNumInferenceSteps] = useState(30);
	
	// State for input values and validation
	const [widthInput, setWidthInput] = useState("512");
	const [heightInput, setHeightInput] = useState("512");
	const [numImagesInput, setNumImagesInput] = useState("1");
	const [inferenceStepsInput, setInferenceStepsInput] = useState("30");
	
	// State for validation errors
	const [widthError, setWidthError] = useState(null);
	const [heightError, setHeightError] = useState(null);
	const [numImagesError, setNumImagesError] = useState(null);
	const [inferenceStepsError, setInferenceStepsError] = useState(null);
	
	// State for API response
	const [generatedImages, setGeneratedImages] = useState([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);
	
	// State for UI display toggles
	const [showDetails, setShowDetails] = useState(false);
	
	/**
	 * Toggle details view mode
	 */
	async function toggleBlackBoxInfo() {
		setShowDetails(!showDetails);
	}
	

	
    /**
     * Handle prompt input change
     * @param {string} value - The new input value
     */
    function handlePromptChange(value) {
        setPrompt(value);
        setError(null);
    }
    
    /**
     * Handle width text input change
     * @param {string} value - The new input value
     */
    function handleWidthChange(value) {
        setWidthInput(value);
        
        const parsedValue = parseInt(value, 10);
        if (isNaN(parsedValue) || value === "") {
            setWidthError("Must be a number");
        } else if (parsedValue < 128) {
            setWidthError("Must be at least 128px");
        } else if (parsedValue > 1024) {
            setWidthError("Must be at most 1024px");
        } else {
            setWidthError(null);
            setWidth(parsedValue);
        }
    }
    
    /**
     * Handle height text input change
     * @param {string} value - The new input value
     */
    function handleHeightChange(value) {
        setHeightInput(value);
        
        const parsedValue = parseInt(value, 10);
        if (isNaN(parsedValue) || value === "") {
            setHeightError("Must be a number");
        } else if (parsedValue < 128) {
            setHeightError("Must be at least 128px");
        } else if (parsedValue > 1024) {
            setHeightError("Must be at most 1024px");
        } else {
            setHeightError(null);
            setHeight(parsedValue);
        }
    }
    
    /**
     * Handle number of images text input change
     * @param {string} value - The new input value
     */
    function handleNumImagesChange(value) {
        setNumImagesInput(value);
        
        const parsedValue = parseInt(value, 10);
        if (isNaN(parsedValue) || value === "") {
            setNumImagesError("Must be a number");
        } else if (parsedValue < 1) {
            setNumImagesError("At least 1 image required");
        } else if (parsedValue > 4) {
            setNumImagesError("Maximum 4 images allowed");
        } else {
            setNumImagesError(null);
            setNumImages(parsedValue);
        }
    }
    
    /**
     * Handle inference steps text input change
     * @param {string} value - The new input value
     */
    function handleInferenceStepsChange(value) {
        setInferenceStepsInput(value);
        
        const parsedValue = parseInt(value, 10);
        if (isNaN(parsedValue) || value === "") {
            setInferenceStepsError("Must be a number");
        } else if (parsedValue < 1) {
            setInferenceStepsError("At least 1 inference step required");
        } else if (parsedValue > 100) {
            setInferenceStepsError("Maximum 100 inference steps allowed");
        } else {
            setInferenceStepsError(null);
            setNumInferenceSteps(parsedValue);
        }
    }
    
    /**
     * Check if there are any validation errors
     * @returns {boolean} True if there are no validation errors
     */
    function hasValidationErrors() {
        return !!(widthError || heightError || numImagesError || inferenceStepsError || !prompt);
    }
	
	/**
	 * Submit the prompt and parameters to the API for image generation
	 */
	async function generateImages() {
		if (!prompt) {
			setError("Please enter a prompt first.");
			return;
		}
		
		setLoading(true);
		setError(null);
		setGeneratedImages([]);
		
		try {
			const requestBody = {
				prompt: prompt,
				height: height,
				width: width,
				num_images: numImages,
				num_inference_steps: numInferenceSteps
			};
			
			const response = await fetch("/invocations", {
				method: "POST",
				headers: {
					"Content-Type": "application/json;charset=UTF-8",
				},
				body: JSON.stringify(requestBody),
			});
			
			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			const jsonResponse = await response.json();

			if (Array.isArray(jsonResponse)) {
				setGeneratedImages(jsonResponse);
			} else if (jsonResponse.images) {
				setGeneratedImages(jsonResponse.images);
			} else if (jsonResponse.predictions && Array.isArray(jsonResponse.predictions)) {
				setGeneratedImages(jsonResponse.predictions);
			} else {
				setGeneratedImages([]);
				throw new Error("Invalid response format from model.");
			}
		} catch (error) {
			console.error("Error when calling the API:", error);
			setError(`Failed to generate images: ${error.message}`);
		} finally {
			setLoading(false);
		}
	}
	

	
	return (
		<div>
			<div className="header">
				<div className="header-logo">
					<img src={iconImage} width="150px" height="150px" alt="Image Generation Logo" /> 
				</div>
				<div className='title-info'>
					<div className="header-title">
						<h3 className='title'>Image Generation with AI Studio</h3>
					</div>
					<div className="header-description">
						<p>Generate customized images based on your text prompt</p>
					</div>
				</div>
			</div>
			
			{/* Prompt Input Card */}
			<Card className="parameter-input-card"
				border="outlined"
				content={
					<div className="outer-padding">
						<h4>Create Images</h4>
						<p>Enter a prompt and adjust parameters to generate images.</p>
						<div className="prompt-input">
							<TextBox
								id="prompt-input"
								label="Image Prompt"
								placeholder="Enter a detailed description of the image you want to generate..."
								value={prompt}
								onChange={handlePromptChange}
								error={!!error}
								helperText={error}
							/>
						</div>
						<div className="input-control input-buttons">
							<div className='input-toggle'>
								<Toggle className="detail-toggle" label="Show Parameters" onChange={setShowDetails} />
							</div>
							<Button 
								className="generate-button" 
								onClick={generateImages}
								disabled={!prompt || loading || hasValidationErrors()}
							>
								{loading ? "Generating..." : "Generate Images"}
							</Button>
						</div>
					</div>
				}
			/>
			
			{/* Main content area with either detailed view or simplified view */}
			{showDetails ? (
				<Card className="white-box"
					border="outlined"
					content={
						<div className="main-detail-box">
							{/* Parameters Card */}
							<Card className="parameters-card"
								border="outlined"
								content={
									<div className='outer-padding'>
										<div className='title-with-icon'>
											<h5>Generation Parameters</h5>
											<div className='title-with-icon-icon'>
												<Tooltip
													arrow
													placement="trailing-end"
													content={
														<span>
															Adjust these parameters to control the image generation process.
															Higher resolution values create larger images but take longer to generate.
														</span>
													}
												>
													<IconInfo 
														size={24}
														ltip
													/>
												</Tooltip>
											</div>
										</div>
										<div className="parameters-container">
											<div className="parameter-sliders">
												<div className="parameter-row">
													<div className="parameter-label">Width:</div>
													<Slider
														className="parameter-slider"
														minValue={128}
														maxValue={1024}
														step={64}
														value={width}
														onChange={(value) => {
															setWidth(value);
															setWidthInput(value.toString());
															setWidthError(null);
														}}
													/>
													<TextBox
														className="parameter-value-input"
														value={widthInput}
														onChange={handleWidthChange}
														suffix="px"
														width={80}
														error={!!widthError}
														helperText={widthError}
													/>
												</div>

												<div className="parameter-row">
													<div className="parameter-label">Height:</div>
													<Slider
														className="parameter-slider"
														minValue={128}
														maxValue={1024}
														step={64}
														value={height}
														onChange={(value) => {
															setHeight(value);
															setHeightInput(value.toString());
															setHeightError(null);
														}}
													/>
													<TextBox
														className="parameter-value-input"
														value={heightInput}
														onChange={handleHeightChange}
														suffix="px"
														width={80}
														error={!!heightError}
														helperText={heightError}
													/>
												</div>

												<div className="parameter-row">
													<div className="parameter-label">Number of Images:</div>
													<Slider
														className="parameter-slider"
														minValue={1}
														maxValue={4}
														step={1}
														value={numImages}
														onChange={(value) => {
															setNumImages(value);
															setNumImagesInput(value.toString());
															setNumImagesError(null);
														}}
													/>
													<TextBox
														className="parameter-value-input"
														value={numImagesInput}
														onChange={handleNumImagesChange}
														width={80}
														error={!!numImagesError}
														helperText={numImagesError}
													/>
												</div>

												<div className="parameter-row">
													<div className="parameter-label">Inference Steps:</div>
													<Slider
														className="parameter-slider"
														minValue={1}
														maxValue={100}
														step={1}
														value={numInferenceSteps}
														onChange={(value) => {
															setNumInferenceSteps(value);
															setInferenceStepsInput(value.toString());
															setInferenceStepsError(null);
														}}
													/>
													<TextBox
														className="parameter-value-input"
														value={inferenceStepsInput}
														onChange={handleInferenceStepsChange}
														width={80}
														error={!!inferenceStepsError}
														helperText={inferenceStepsError}
													/>
												</div>
											</div>
										</div>
									</div>
								}
							/>
							
							{/* Generated Images Card */}
							<Card className="parameters-card"
								border="outlined"
								content={
									<div className='outer-padding'>
										<div className='title-with-icon'>
											<h5>Generated Images</h5>
											<div className='title-with-icon-icon'>
												<Tooltip
													arrow
													placement="trailing-end"
													content={
														<span>
															These are your AI-generated images based on the provided prompt and parameters.
															The quality and style depend on the prompt details and inference steps.
														</span>
													}
												>
													<IconInfo 
														size={24}
														ltip
													/>
												</Tooltip>
											</div>
										</div>
										<div className="images-container">
											{loading ? (
												<div className="loading-container">
													<ProgressIndicator
                                                        appearance="linear"
                                                        behavior="indeterminate"
                                                        size="large"
                                                    />
												</div>
											) : generatedImages.length > 0 ? (
												<div className="image-grid">
													{generatedImages.map((image, index) => (
														<div key={index} className="image-card">
															<img 
																className="generated-image"
																src={typeof image === 'string' ? image : image.url || image.data}
																alt={`Generated image ${index + 1} for prompt: ${prompt}`}
															/>
														</div>
													))}
												</div>
											) : (
												<p>No images generated yet. Enter a prompt and click "Generate Images" to start.</p>
											)}
										</div>
									</div>
								}
							/>
						</div>
					}
				/>
			) : (
				<div>
					<Card className="black-box"
						border="outlined"
						content={
							<div>
								<div className='title-with-icon' style={{display:"flex", justifyContent:"center"}}>
									<h4>Image Generation</h4>
									<div className='title-with-icon-icon'>
										<Tooltip
											arrow
											placement="trailing-end"
											content={
												<span>
													Toggle between simplified and detailed view modes
													for more control over image generation parameters.
												</span>
											}
										>
											<IconInfo 
												size={24} 
												onClick={toggleBlackBoxInfo} 
												filled={showDetails}
												ltip
											/>
										</Tooltip>
									</div>
								</div>
								<div>
									{error && <p className="error-message">{error}</p>}
								</div>
								<div className='outer-padding'>
									{loading ? (
										<div className="loading-container">
                                            <ProgressIndicator
                                                appearance="linear"
                                                behavior="indeterminate"
                                                size="large"
                                            />
										</div>
									) : generatedImages.length > 0 ? (
										<div className="image-grid">
											{generatedImages.map((image, index) => (
												<div key={index} className="image-card">
													<img
														className="generated-image"
														src={typeof image === 'string' ? image : image.url || image.data}
														alt={`Generated image ${index + 1}`}
													/>
												</div>
											))}
										</div>
									) : (
										<p>Enter a prompt and click "Generate Images" to create images.</p>
									)}
								</div>
							</div>
						}
					/>
				</div>
			)}
		</div>
	);
}

export default App;

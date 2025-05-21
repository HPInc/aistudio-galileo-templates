import httpcore
import logging
import time
import concurrent.futures
from functools import partial
from tqdm import tqdm

class LLMContextUpdater:
    def __init__(self, llm_chain, prompt_template, verbose=False, print_prompt=False, overwrite=True,
                item_timeout=30, max_retries=2, batch_size=None):
        """
        :param llm_chain: LLM chain object with an .invoke() method
        :param prompt_template: PromptTemplate used to render the final prompt
        :param verbose: If True, enable logging output
        :param print_prompt: If True, print the formatted prompt
        :param overwrite: If True, always overwrite context even if it exists
        :param item_timeout: Timeout in seconds for processing each item
        :param max_retries: Maximum number of retries for failed items
        :param batch_size: If provided, process data in batches of this size
        """
        self.llm_chain = llm_chain
        self.prompt_template = prompt_template
        self.print_prompt = print_prompt
        self.overwrite = overwrite
        self.item_timeout = item_timeout
        self.max_retries = max_retries
        self.batch_size = batch_size

        # Configure logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
        
        # Optional timeout for the entire update process
        self.global_timeout = None

    def update(self, data_structure, global_timeout=None):
        """
        Update context for all items in data_structure.
        
        Args:
            data_structure: List of dictionaries containing code snippets and metadata
            global_timeout: Optional timeout for the entire update process in seconds
            
        Returns:
            Updated list of dictionaries with generated context
        """
        self.global_timeout = global_timeout
        start_time = time.time()
        updated_structure = []
        
        # Skip files that are too large - they are likely to cause timeouts
        filtered_data = []
        for item in data_structure:
            code = item.get('code', '')
            if len(code) > 50000:  # Skip very large files (>50KB)
                filename = item.get('filename', 'unknown')
                self.logger.warning(f"Skipping large file {filename} ({len(code)} bytes)")
                item['context'] = f"File: {filename} (too large for processing: {len(code)} bytes)"
                updated_structure.append(item)
            else:
                filtered_data.append(item)
        
        data_structure = filtered_data
        
        # Early exit if no items to process after filtering
        if not data_structure:
            self.logger.warning("No files to process after filtering large files")
            return updated_structure
        
        # Batch processing if batch_size is specified
        if self.batch_size and len(data_structure) > self.batch_size:
            self.logger.info(f"Processing {len(data_structure)} items in batches of {self.batch_size}")
            
            # Process data in batches
            for i in range(0, len(data_structure), self.batch_size):
                # Check if we've exceeded the global timeout
                elapsed_time = time.time() - start_time
                remaining_time = None if self.global_timeout is None else self.global_timeout - elapsed_time
                
                if self.global_timeout and elapsed_time > self.global_timeout:
                    self.logger.warning(f"Global timeout ({self.global_timeout}s) reached after processing {len(updated_structure)} items")
                    # Add remaining items with basic context
                    remaining_items = len(data_structure) - len(updated_structure)
                    if remaining_items > 0:
                        self.logger.info(f"Adding {remaining_items} remaining items with basic context")
                        for item in data_structure[len(updated_structure):]:
                            filename = item.get('filename', 'unknown')
                            if 'context' not in item or not item['context'] or self.overwrite:
                                item['context'] = f"File: {filename} (processing timed out)"
                            updated_structure.append(item)
                    return updated_structure
                
                # Process the current batch
                batch = data_structure[i:i+self.batch_size]
                batch_desc = f"Batch {i//self.batch_size + 1}/{(len(data_structure) + self.batch_size - 1)//self.batch_size}"
                
                # Adjust item timeout based on remaining time if applicable
                if remaining_time is not None:
                    # Ensure each item gets a fair share of the remaining time, with a minimum threshold
                    items_left = len(data_structure) - i
                    adjusted_timeout = max(3, min(self.item_timeout, remaining_time / items_left * 0.8))
                    self.item_timeout = adjusted_timeout
                
                batch_results = self._process_batch(batch, batch_desc)
                updated_structure.extend(batch_results)
                
                # Optional small delay between batches to prevent resource exhaustion
                time.sleep(0.5)
        else:
            # Process everything as a single batch
            updated_structure.extend(self._process_batch(data_structure, "Updating Contexts"))
            
        return updated_structure

    def _process_batch(self, batch, desc):
        """Process a batch of items with progress tracking."""
        results = []
        batch_start_time = time.time()
        
        for i, item in enumerate(batch):
            # Check if we need to abort the batch due to global timeout
            if self.global_timeout:
                elapsed_total = time.time() - batch_start_time
                if elapsed_total > self.global_timeout * 0.8:  # If we've used 80% of our time
                    remaining = len(batch) - i
                    self.logger.warning(f"Batch processing taking too long. Skipping remaining {remaining} items")
                    
                    # Add the remaining items with basic context
                    for j in range(i, len(batch)):
                        skipped_item = batch[j]
                        filename = skipped_item.get('filename', 'unknown')
                        if 'context' not in skipped_item or not skipped_item['context'] or self.overwrite:
                            skipped_item['context'] = f"File: {filename} (skipped due to batch timeout)"
                        results.append(skipped_item)
                    
                    return results
            
            # Show progress indicator
            if i > 0:  # After the first item, we can estimate time
                elapsed = time.time() - batch_start_time
                avg_time = elapsed / i
                est_remaining = avg_time * (len(batch) - i)
                if hasattr(tqdm, '_instances') and tqdm._instances:
                    for t in tqdm._instances:
                        t.set_description(f"{desc} (est. {est_remaining:.1f}s left)")
            
            # Try to process the item with retries
            processed = False
            retries = 0
            
            while not processed and retries <= self.max_retries:
                try:
                    self._process_item(item)
                    processed = True
                except Exception as e:
                    retries += 1
                    if retries <= self.max_retries:
                        self.logger.warning(f"Retry {retries}/{self.max_retries} for {item.get('filename', 'unknown')}: {str(e)}")
                        time.sleep(1)  # Small delay before retry
                    else:
                        self.logger.error(f"Failed after {self.max_retries} retries for {item.get('filename', 'unknown')}")
                        # Add basic context for failed items
                        if 'context' not in item or not item['context'] or self.overwrite:
                            item['context'] = f"File: {item.get('filename', 'unknown')} (processing failed after {self.max_retries} retries)"
            
            results.append(item)
            
        return results

    def _process_item(self, item):
        """Process a single item with timeout protection."""
        code = item['code']
        filename = item['filename']
        context = item.get('context', '')

        # Skip if context exists and we're not overwriting
        if context and not self.overwrite:
            self.logger.debug(f"Skipping context for: {filename}")
            return
        
        # Skip very large files - they are likely to cause timeouts
        if len(code) > 50000:  # Skip files > 50KB
            item['context'] = f"File: {filename} (too large for processing: {len(code)} bytes)"
            self.logger.warning(f"Skipping large file {filename} ({len(code)} bytes)")
            return
            
        # For very large files (10KB-50KB), truncate the code to avoid long processing times
        truncated = False
        if len(code) > 10000:
            truncated_length = 10000
            code = code[:truncated_length] + f"\n\n... (truncated, original size: {len(code)} bytes)"
            truncated = True
            self.logger.info(f"Truncated large file {filename} from {len(item['code'])} to {truncated_length} bytes")
        
        # Prepare inputs for the LLM
        inputs = {
            "code": code,
            "filename": filename,  
            "context": context
        }

        # Format the prompt and optionally print it
        rendered_prompt = self.prompt_template.format(**inputs)
        if self.print_prompt:
            self.logger.debug(f"\nPrompt for file {filename}:\n{rendered_prompt}\n{'=' * 60}")

        # Process with timeout guard using concurrent.futures
        try:
            # Invoke the LLM with the rendered prompt string, not the raw inputs dict
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.llm_chain.invoke, rendered_prompt)
                response = future.result(timeout=self.item_timeout)
                
                # Update the context with the LLM response
                if truncated:
                    item['context'] = f"{response.strip()} (Note: analysis based on truncated file)"
                else:
                    item['context'] = response.strip()
                self.logger.debug(f"Context updated for: {filename}")
                
        except concurrent.futures.TimeoutError:
            self.logger.warning(f"Timeout ({self.item_timeout}s) for {filename}")
            item['context'] = f"File: {filename} (LLM processing timed out after {self.item_timeout}s)"
            raise
        except httpcore.ConnectError as e:
            self.logger.error(f"Connection error on {filename}: {str(e)}")
            item['context'] = f"File: {filename} (Connection error: {str(e)[:50]})"
            raise
        except httpcore.ProtocolError as e:
            self.logger.error(f"Protocol error on {filename}: {str(e)}")
            item['context'] = f"File: {filename} (Protocol error: {str(e)[:50]})"
            raise
        except Exception as e:
            self.logger.error(f"Error processing {filename}: {str(e)}")
            item['context'] = f"File: {filename} (Error: {str(e)[:50]})"
            raise

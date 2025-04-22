import os
import logging
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer, util

class UltraFeedbackVisualizer:
    """
    Logs human feedback data from the UltraFeedback dataset into TensorBoard
    for better understanding and presentation of preferred vs. rejected model responses.

    It uses `score_chosen` and `score_rejected` to plot charts and markdown summaries
    that are easy to interpret, even by non-technical audiences.

    Attributes:
        train_dataset (Dataset): Subset of the UltraFeedback training data.
        test_dataset (Dataset): Subset of the UltraFeedback test data.
        log_dir (str): Path to the TensorBoard log directory.
        max_samples (int): Maximum number of examples to visualize.
    """
    
    def __init__(self, train_dataset, test_dataset, log_dir="/phoenix/tensorboard/tensorlogs", max_samples=20):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.log_dir = log_dir
        self.max_samples = max_samples

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("UltraFeedbackVisualizer")

    def _extract_text(self, field, field_name="unknown", idx=0):
        """
        Safely extracts and normalizes text content from fields that may be strings, dicts, or lists.

        Args:
            field (Union[str, dict, list]): The text or structured field.
            field_name (str): Field label (for logging).
            idx (int): Example index (for debugging).

        Returns:
            str: Normalized text string.
        """
        try:
            if isinstance(field, dict) and "text" in field:
                return field["text"]
            elif isinstance(field, list):
                return " ".join(
                    [f["text"] if isinstance(f, dict) and "text" in f else str(f) for f in field]
                )
            else:
                return str(field)
        except Exception as e:
            self.logger.error(f"[Example {idx}] ❌ Error parsing field '{field_name}': {e}")
            return "[ERROR]"

    def log_dataset(self, dataset, tag_prefix="train"):
        """
        Logs a subset of dataset examples into TensorBoard as text, scalar scores, and differences.

        Args:
            dataset (Dataset): The dataset split to log (train or test).
            tag_prefix (str): TensorBoard tag namespace (e.g., "train" or "test").
        """
        score_chosen_list = []
        score_rejected_list = []
        score_delta_list = []

        for idx, example in enumerate(dataset.select(range(min(len(dataset), self.max_samples)))):
            try:
                prompt = self._extract_text(example["prompt"], "prompt", idx)
                chosen = self._extract_text(example["chosen"], "chosen", idx)
                rejected = self._extract_text(example["rejected"], "rejected", idx)
                score_chosen = example.get("score_chosen", 0.0)
                score_rejected = example.get("score_rejected", 0.0)
                delta = score_chosen - score_rejected

                score_chosen_list.append(score_chosen)
                score_rejected_list.append(score_rejected)
                score_delta_list.append(delta)

                # 🔢 Log the delta score (how much better the chosen response was)
                self.writer.add_scalar(f"{tag_prefix}/ScoreDelta/Example_{idx}", delta, global_step=idx)

                # 📈 Log chosen/rejected raw scores
                self.writer.add_scalar(f"{tag_prefix}/Score/Chosen", score_chosen, global_step=idx)
                self.writer.add_scalar(f"{tag_prefix}/Score/Rejected", score_rejected, global_step=idx)

                # 📝 Pretty markdown summary to visualize the result
                preferred_tag = "🟢 Chosen" if delta >= 0 else "🔴 Rejected"
                markdown_text = f"""
### ❓ Prompt:
{prompt}

---

### 🟢 Chosen (score: {score_chosen:.1f}):
{chosen}

---

### 🔴 Rejected (score: {score_rejected:.1f}):
{rejected}

---

🏁 **Humans preferred:** {preferred_tag}  
📉 Score difference: **{delta:.2f} points**
"""

                self.writer.add_text(f"{tag_prefix}/example_{idx}/Visual", markdown_text, global_step=idx)
                self.logger.info(f"[Example {idx}] ✅ Logged successfully")

            except Exception as e:
                self.logger.error(f"[Example {idx}] ❌ Failed to process example: {e}")

        # 📊 Log mean summary scores for all samples
        if score_chosen_list:
            self.writer.add_scalar(f"summary/{tag_prefix}_mean_chosen", sum(score_chosen_list) / len(score_chosen_list))
        if score_rejected_list:
            self.writer.add_scalar(f"summary/{tag_prefix}_mean_rejected", sum(score_rejected_list) / len(score_rejected_list))
        if score_delta_list:
            self.writer.add_scalar(f"summary/{tag_prefix}_mean_delta", sum(score_delta_list) / len(score_delta_list))

    def run(self):
        """
        Executes the full logging pipeline for both train and test splits.
        """
        self.logger.info("📊 Logging training samples (human feedback only)...")
        self.log_dataset(self.train_dataset, tag_prefix="train")

        self.logger.info("📊 Logging test samples (human feedback only)...")
        self.log_dataset(self.test_dataset, tag_prefix="test")

        self.writer.close()
        self.logger.info(f"✅ Human feedback exploration complete!\n"
                         f"Launch TensorBoard with:\n"
                         f"tensorboard --logdir={self.log_dir} --port 6006")

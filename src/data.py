"""
data.py - LIBERO-Spatial training dataloader.

HYBRID: we reuse the parent repo's battle-tested RLDS/TFDS pipeline verbatim (chunking,
bounds_q99 normalization, image aug, the action-token + prompt construction). This file just
wires those pieces together with the Phase-1 settings and yields the collated batch dict:
    input_ids, attention_mask, pixel_values (B, 12, 224, 224), labels, actions (B,8,7), proprio.
"""
import constants as C                                                    # noqa: F401
from torch.utils.data import DataLoader                                  # noqa: E402
from prismatic.models.backbones.llm.prompting import PurePromptBuilder   # noqa: E402
from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer               # noqa: E402
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset       # noqa: E402


def build_dataloader(processor, batch_size=C.BATCH_SIZE,
                     shuffle_buffer_size=C.SHUFFLE_BUFFER_SIZE, image_aug=True):
    """Returns (dataloader, dataset_statistics). `processor` is from model.build_processor()."""
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=(C.NUM_IMAGES > 1),
        use_proprio=True,
        use_minivlm=True,
    )

    dataset = RLDSDataset(
        C.DATA_ROOT_DIR,
        C.DATASET_NAME,
        batch_transform,
        resize_resolution=tuple(C.IMAGE_RESOLUTION),
        shuffle_buffer_size=shuffle_buffer_size,
        image_aug=image_aug,
        train=True,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=None, 
        collate_fn=collator,
        num_workers=0,
    )
    return dataloader, dataset.dataset_statistics

# Backup and cleanup status

## Verified released model

A fresh, unauthenticated download of the pinned Hugging Face model revision passed every bundled SHA-256 check. The standalone CLI loaded and generated outside the original project in a separate environment using MLX 0.32.2, mlx-metal 0.32.2, and tokenizers 0.22.2 installed from the published requirements. See [verification](provenance/CLEAN_DOWNLOAD_VERIFICATION.json).

## Preserved records

- [Checkpoint inventory](provenance/CHECKPOINT_INVENTORY.json): original local bundles and recorded file identities. Inventory is not proof of off-machine backup.
- [Training measurements](results/training_history/): numeric records from Unified384, Repair512, and the selected continuation. Logs cover each entire configured run; the retained boundaries are +384, +128, and +256 updates. Later measurements do not belong to the selected checkpoint's exposure.
- [Training archive manifest](provenance/TRAINING_RECORDS_MANIFEST.json): hashes of 168 selected recipe/input/log/evaluation files prepared for separate backup. The archive is now hosted on Hugging Face and all 168 inner files have been checked after a fresh download. See [hosted-backup verification](provenance/HOSTED_BACKUP_VERIFICATION.json).

## Cleanup rule

Only remove a potentially useful file after its hosted copy is identified and verified. Keep original source, uncommitted work, tokenizer, prepared data, checkpoint metadata and logs until their intended backups are complete. The live playground's selected checkpoint must remain available or be explicitly migrated.

The public inference release excludes optimizer state and earlier model checkpoints. Those files cannot be treated as backed up merely because the selected model is downloadable. The original 62-bundle inventory includes other experiments that are outside the released model lineage.

Any removal receipt should record exact paths, hashes, sizes and verified hosted destinations. Cleanup is not a full workspace wipe unless all required assets have verified backups.

## Restore archived training inputs

Download [the selected training record archive](https://huggingface.co/wzebrowski/OpenSML-150M/blob/main/OpenSML-150M-training-records.tar.gz) and [its manifest](https://huggingface.co/wzebrowski/OpenSML-150M/blob/main/TRAINING_RECORDS_MANIFEST.json). The archive preserves original workspace-relative V2 names, independently of the V1 release naming. Verify the compressed SHA-256 and each required member before restoring. Extract into a separate directory first, then restore selected files to their original paths; do not overwrite active work blindly.

The archive contains selected SFT prepared source pools, experiment recipes, logs and evaluation records. It is not a complete pretraining-data backup and excludes every optimizer/model tensor.

## Completed local cleanup

Seven prepared training-input copies and two duplicate model-release exports were removed only after hosted content verification. About 1.4 GiB was reclaimed. The selected checkpoint serving the local playground, every other model checkpoint, optimizer tensors, code and logs remain local. Exact removed paths and hosted restore locations are in [the cleanup receipt](provenance/CLEANUP_RECEIPT.json). This is a verified limited cleanup, not a complete workspace wipe.

To verify a downloaded archive without extracting it:

```bash
python scripts/verify_training_archive.py --archive OpenSML-150M-training-records.tar.gz --manifest TRAINING_RECORDS_MANIFEST.json
```

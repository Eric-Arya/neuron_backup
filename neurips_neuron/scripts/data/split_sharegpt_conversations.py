"""
This script is largely copied from the Vicuna repo: https://github.com/lm-sys/FastChat/blob/main/fastchat/data/split_long_conversation.py
We fixed a bug in `split_one_sample`, which previously includes long conversations in the processed data. Now we skip these long conversations.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
import time
import transformers
from tqdm import tqdm


def make_sample(sample, start_idx, end_idx):
    assert (end_idx - start_idx) % 2 == 0
    return {
        "id": sample["id"] + "_" + str(start_idx),
        "conversations": sample["conversations"][start_idx:end_idx],
    }


tokenizer = max_length = None


def split_one_sample(sample):
    conversations = sample["conversations"]
    # A training example must consist of complete human/gpt rounds.  Do not
    # truncate an odd conversation here: silently dropping its final turn can
    # turn malformed data into a misleadingly valid example.
    if len(conversations) < 2 or len(conversations) % 2 != 0:
        return []
    if any(
        message.get("from") != ("human" if i % 2 == 0 else "gpt")
        for i, message in enumerate(conversations)
    ):
        return []

    tokenized_lens = []
    for c in conversations:
        length = len(tokenizer(c["value"]).input_ids) + 6
        tokenized_lens.append(length)

    start_idx = None
    cur_len = 0
    new_samples = []
    for i in range(0, len(conversations), 2):
        tmp_len = tokenized_lens[i] + tokenized_lens[i + 1]
        if tmp_len > max_length:
            # There is no valid way to split inside a user/assistant round.
            # Flush the preceding chunk, then skip this oversized round.
            if start_idx is not None:
                new_samples.append(make_sample(sample, start_idx, i))
            start_idx = None
            cur_len = 0
            continue

        if start_idx is None:
            start_idx = i
            cur_len = tmp_len
        elif cur_len + tmp_len <= max_length:
            cur_len += tmp_len
        else:
            new_samples.append(make_sample(sample, start_idx, i))
            start_idx = i
            cur_len = tmp_len

    if start_idx is not None:
        new_samples.append(make_sample(sample, start_idx, len(conversations)))

    return new_samples


def split_all(
    content,
    begin,
    end,
    tokenizer_,
    max_length_,
    num_workers=None,
    progress_every=1000,
    heartbeat_seconds=30,
):
    """
    Keep the maximum round of conversations within the max token length constraint
    """
    global tokenizer, max_length
    tokenizer = tokenizer_
    max_length = max_length_

    content = content[begin:end]
    new_content = []

    max_workers = num_workers or min(32, os.cpu_count() or 1)
    if max_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if heartbeat_seconds <= 0:
        raise ValueError("--heartbeat-seconds must be positive")

    total = len(content)
    print(
        f"Starting tokenization: {total} records, {max_workers} workers, "
        f"max_length={max_length}",
        flush=True,
    )
    started_at = time.monotonic()
    last_report_at = started_at
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        print("Submitting records to workers...", flush=True)
        pending = set()
        next_index = 0
        queue_limit = max_workers * 4
        while next_index < total and len(pending) < queue_limit:
            pending.add(executor.submit(split_one_sample, content[next_index]))
            next_index += 1
        print(
            f"Submitted initial batch of {len(pending)} records; collecting results...",
            flush=True,
        )
        progress = tqdm(total=total, desc="Splitting ShareGPT", unit="record")
        while pending:
            done, pending = wait(
                pending,
                timeout=heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                elapsed = time.monotonic() - started_at
                print(
                    f"Heartbeat: {completed}/{total} records complete, "
                    f"{len(pending)} pending, elapsed={elapsed:.0f}s; "
                    "workers are still processing",
                    flush=True,
                )
                continue
            for future in done:
                new_content.extend(future.result())
                completed += 1
                progress.update(1)
                if next_index < total:
                    pending.add(executor.submit(split_one_sample, content[next_index]))
                    next_index += 1
                now = time.monotonic()
                if completed % progress_every == 0 or completed == total or now - last_report_at >= heartbeat_seconds:
                    elapsed = now - started_at
                    rate = completed / elapsed if elapsed else 0.0
                    print(
                        f"Progress: {completed}/{total} records, "
                        f"{len(new_content)} chunks, {rate:.1f} records/s, "
                        f"pending={len(pending)}",
                        flush=True,
                    )
                    last_report_at = now
        progress.close()

    return new_content


def filter_invalid_roles(content):
    new_content = []
    for i, c in enumerate(content):
        roles = ["human", "gpt"]
        if len(c["conversations"]) <= 0:
            continue

        valid = True
        for j, s in enumerate(c["conversations"]):
            if s["from"] != roles[j % 2]:
                valid = False
                break

        if valid:
            new_content.append(c)

    return new_content


def main(args):
    if not args.in_files:
        raise ValueError("--in-files must contain at least one JSON file")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    print("Loading input JSON files...", flush=True)
    content = []
    for file in args.in_files:
        started_at = time.monotonic()
        with open(file) as fin:
            records = json.load(fin)
        content.extend(records)
        print(
            f"Loaded {file}: {len(records)} records in "
            f"{time.monotonic() - started_at:.1f}s",
            flush=True,
        )
    print(f"Total loaded records: {len(content)}", flush=True)
    print("Loading tokenizer...", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )
    print("Tokenizer loaded.", flush=True)
    new_content = split_all(
        content,
        args.begin,
        args.end,
        tokenizer,
        args.max_length,
        args.num_workers,
        args.progress_every,
        args.heartbeat_seconds,
    )
    print("Filtering invalid output records...", flush=True)
    new_content = filter_invalid_roles(new_content)

    print(f"total: {len(content)}, new: {len(new_content)}")
    with open(args.out_file, "w") as fout:
        json.dump(new_content, fout, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-files", nargs="+", type=str)
    parser.add_argument("--out-file", type=str, default="sharegpt_split.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of tokenizer worker processes (default: min(32, CPU count))",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print a progress summary every N completed records (default: 1000)",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="Print progress even when fewer than --progress-every records finish (default: 30)",
    )
    args = parser.parse_args()
    main(args)

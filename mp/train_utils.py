import os
import torch
import torch.distributed as dist
from datetime import datetime
import tqdm
from pytorch_transformers import BertTokenizer, BertForMaskedLM
from bert import BertForSequenceClassificationMP

g_gigabyte = 1024**3

def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run



def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None, global_step=None):
    model.train()
    fsdp_loss = torch.zeros(2).to(0)
  
    if sampler:
        sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for step, batch in enumerate(train_loader):
        global_step[0] += 1
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                  'labels':         batch[3]}
        output = model(**inputs)
        loss = output[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank==0:
            inner_pbar.update(1)

    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy


def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    fsdp_loss = torch.zeros(2).to(0)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(t.to(0) for t in batch)
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],  # XLM don't use segment_ids
                    'labels':         batch[3]}
            output = model(**inputs)
            fsdp_loss[0] += output[0].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    # dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def setup_model(model_name):
        model = BertForSequenceClassificationMP.from_pretrained(model_name)
        tokenizer =  BertTokenizer.from_pretrained(model_name)
        return model, tokenizer

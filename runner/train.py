import torch
import logging
from models.loaders.dataloaders import get_loader

log = logging.getLogger(__name__)


def dpr_loss(sim_matrix):
    log_softmax = torch.log_softmax(sim_matrix, dim=1)
    loss = -torch.mean(torch.diagonal(log_softmax))
    return loss


def train_one_epoch(
        model, train_loader, optimizer,
        device
):
    model.train()
    total_loss = 0
    step_count = 0
    for batch in train_loader:
        # batch {"question_inputs", "passage_inputs"}
        question_embeddings, passage_embeddings = model(
            batch["q_inputs"].to(device), batch["p_inputs"].to(device))
        sim_matrix = torch.matmul(question_embeddings, passage_embeddings.T)
        loss = dpr_loss(sim_matrix)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        step_count += 1
        # log.debug(f"Batch Loss: {loss.item():.4f}")
    if step_count == 0:
        return 0.0
    avg_loss = total_loss / step_count
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    step_count = 0
    for batch in val_loader:
        question_embeddings, passage_embeddings = model(
            batch["q_inputs"].to(device), batch["p_inputs"].to(device))
        sim_matrix = torch.matmul(question_embeddings, passage_embeddings.T)
        loss = dpr_loss(sim_matrix)
        total_loss += loss.item()
        step_count += 1
    if step_count == 0:
        return 0.0
    avg_loss = total_loss / step_count
    return avg_loss


@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    ranks = None
    for batch in test_loader:
        question_embeddings, passage_embeddings = model(
            batch["q_inputs"].to(device), batch["p_inputs"].to(device))
        sim_matrix = torch.matmul(question_embeddings, passage_embeddings.T)
        if ranks is None:
            ranks = [0] * sim_matrix.size(0)
        target_scores = torch.diagonal(sim_matrix).unsqueeze(1)
        is_larger = sim_matrix > target_scores
        ranks = torch.sum(is_larger, dim=1) + 1  # rank starts from 1
        for rank in ranks.cpu().numpy():
            ranks[rank - 1] += 1
    return ranks



def train_loop(
        epochs, model, train_loader,
        val_loader, test_loader, optimizer, 
        scheduler, device, patience=10, 
        early_stopping=True
):

    # 조기종료를 위한 변수
    best_val_loss = float('inf')
    wait = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            device
        )
        val_loss = validate(
            model, val_loader, device
        )
        log.info(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        if scheduler:
            scheduler.step()

        # 조기 종료
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    log.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

    return {
        "test": test(model, test_loader, device), 
        "best_val_loss": best_val_loss, 
        "epochs": epoch+1
        }

def train(
        model, epochs, loader_config,
        optimizer_config, scheduler_config, device,
        patience=10, early_stopping=True
):
    train_loader = get_loader(**loader_config['train'])
    val_loader = get_loader(**loader_config['val'])
    test_loader = get_loader(**loader_config['test'])

    optimizer = getattr(torch.optim, optimizer_config['type'])(
        model.parameters(), **optimizer_config['params']
    )

    scheduler = getattr(torch.optim.lr_scheduler, scheduler_config['type'])(
        optimizer, **scheduler_config['params']
    ) if scheduler_config else None 

    return train_loop(
        epochs, model, train_loader,
        val_loader, test_loader, optimizer,
        scheduler, device, patience,
        early_stopping
    )
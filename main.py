import torch
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from models.DPR import build_dpr_model
from runner.train import train

log = logging.getLogger(__name__)
results = []  # multi run시 결과 한눈에 보기 위해 사용

#재현성을 위한 seed 고정
def set_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시
    np.random.seed(seed)
    random.seed(seed)
    # 결정론적 연산을 위한 설정 (필요 시)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


@hydra.main(config_path="configs", version_base=None)
def run(config):
    device = torch.device(config.device)
    log.info(f"Using device: {device}")
    set_seed(config.seed)

    model = build_dpr_model(
        Q_encoder_config=config.model.Q_encoder,
        P_encoder_config=config.model.P_encoder,
        device=device,
        drop_out=config.model.drop_out
    ) 
    log.info("Model built successfully.")
    result = train(
        model=model,
        loader_config=config.loader,
        **config.train,
        device=device)

    out_put_dir = HydraConfig.get().runtime.output_dir  # 모델 저장할 때 사용
    model_path = f"{out_put_dir}/final_model.pth"
    torch.save(model.state_dict(), model_path)
    log.info(f"Model saved to {model_path}")

    results.append(result)


if __name__ == "__main__":
    run()
    log.info(results)

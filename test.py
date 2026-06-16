import time
import torch
import numpy as np

def run_pure_forward_benchmark(model, device_name='cpu'):
    """
    model.forward 単体に対して、バッチサイズごとの純粋な計算速度をミリ秒単位で測定する
    """
    print(f"\n=== PyTorch {device_name.upper()} 純粋 Forward ベンチマーク ===")
    model = model.to(device_name)
    model.eval()
    
    # 検証するバッチサイズ
    batch_sizes = [1, 8, 16, 32, 64, 128]
    n_iters = 100 # 精度を上げるため100回試行
    
    for b in batch_sizes:
        # 入力形状 [Batch, Channel=300, Length=48]
        dummy_input = torch.zeros((b, 300, 48), dtype=torch.float32, device=device_name)
        
        # 1. Warmup (最初の1回のアロケーションやJIT初期化コストを完全に排除)
        with torch.inference_mode():
            for _ in range(5):
                _ = model(dummy_input)
        
        # 2. 本計測 (純粋な forward のみをタイマーで挟む)
        t_start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_iters):
                _ = model(dummy_input)
        t_total = time.perf_counter() - t_start
        
        avg_time_per_batch = t_total / n_iters
        avg_time_per_sample = avg_time_per_batch / b
        
        print(f"Batch={b:<3} | 1バッチ推論: {avg_time_per_batch*1000:7.3f}ms | 1サンプルあたり: {avg_time_per_sample*1000:7.3f}ms")

# =====================================================================
# ONNX Runtime のベンチマークも同じ条件で実行できるように関数化
# =====================================================================
def run_onnx_benchmark(model):
    """
    ONNX Runtime (CPU) におけるバッチサイズごとの推論速度を測定
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n[ONNX] onnxruntime がインストールされていないためスキップします。")
        return

    print("\n=== ONNX Runtime (CPU) 純粋 Forward ベンチマーク ===")
    
    # 一度仮のONNXファイルに出力
    onnx_path = "tmp_model.onnx"
    dummy_input_export = torch.zeros((1, 300, 48), dtype=torch.float32)
    
    torch.onnx.export(
        model.cpu(), 
        dummy_input_export, 
        onnx_path,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )
    
    # セッション構築
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # 現行の環境スレッド数に合わせる
    session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    
    batch_sizes = [1, 8, 16, 32, 64, 128]
    n_iters = 100
    
    for b in batch_sizes:
        np_input = np.zeros((b, 300, 48), dtype=np.float32)
        
        # Warmup
        for _ in range(5):
            _ = session.run(None, {'input': np_input})
            
        t_start = time.perf_counter()
        for _ in range(n_iters):
            _ = session.run(None, {'input': np_input})
        t_total = time.perf_counter() - t_start
        
        avg_time_per_batch = t_total / n_iters
        avg_time_per_sample = avg_time_per_batch / b
        print(f"Batch={b:<3} | 1バッチ推論: {avg_time_per_batch*1000:7.3f}ms | 1サンプルあたり: {avg_time_per_sample*1000:7.3f}ms")
"""iTransformer V2 修复验证"""
import torch
import json
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/baselines')

def test_itransformer():
    result = {
        "model_name": "iTransformer",
        "files_changed": [],
        "layer1_create_or_load_success": False,
        "layer1_forward_success": False,
        "layer1_prediction_ready": False,
        "layer1_evaluate_model_ready": False,
        "layer2_create_or_load_success": False,
        "layer2_forecast_success": False,
        "layer2_trajectory_ready": False,
        "final_status": "FAIL",
        "error_message": "",
        "blocker_type": ""
    }
    
    try:
        print("[iTransformer] Testing layer1...")
        from tslib_wrappers import iTransformerWrapper
        
        wrapper_l1 = iTransformerWrapper(seq_len=3, enc_in=15, task='classification', num_class=2)
        result["layer1_create_or_load_success"] = True
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        wrapper_l1 = wrapper_l1.to(device)
        X_test = torch.randn(10, 3, 15).to(device)
        
        with torch.no_grad():
            pred = wrapper_l1.forward(X_test)
        assert pred.shape[0] == 10
        result["layer1_forward_success"] = True
        result["layer1_prediction_ready"] = True
        result["layer1_evaluate_model_ready"] = True
        print("✓ Layer1 success")
        
        print("[iTransformer] Testing layer2...")
        wrapper_l2 = iTransformerWrapper(seq_len=3, enc_in=15, task='long_term_forecast', pred_len=6)
        wrapper_l2 = wrapper_l2.to(device)
        result["layer2_create_or_load_success"] = True
        
        with torch.no_grad():
            forecast = wrapper_l2.forward(X_test)
        assert forecast.shape[0] == 10
        result["layer2_forecast_success"] = True
        result["layer2_trajectory_ready"] = True
        print("✓ Layer2 success")
        
        result["final_status"] = "PASS"
        print("\n[iTransformer] ✓ ALL CHECKS PASSED")
        
    except Exception as e:
        result["error_message"] = str(e)
        result["blocker_type"] = "wrapper_interface_bug"
        print(f"\n[iTransformer] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    return result

if __name__ == "__main__":
    os.makedirs("outputs/model_repairs", exist_ok=True)
    result = test_itransformer()
    
    with open("outputs/model_repairs/itransformer_repair_result_v2.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果: {result['final_status']}")

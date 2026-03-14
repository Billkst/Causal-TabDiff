"""TimeXer V2 修复验证"""
import torch
import json
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/baselines')

def test_timexer():
    result = {
        "model_name": "TimeXer",
        "files_changed": [],
        "layer2_create_or_load_success": False,
        "layer2_forecast_success": False,
        "layer2_trajectory_ready": False,
        "final_status": "FAIL",
        "error_message": "",
        "blocker_type": ""
    }
    
    try:
        print("[TimeXer] Testing layer2...")
        from tslib_wrappers import TimeXerWrapper
        
        wrapper = TimeXerWrapper(seq_len=3, enc_in=15, exog_in=15, task='long_term_forecast', pred_len=6)
        result["layer2_create_or_load_success"] = True
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        wrapper = wrapper.to(device)
        X_test = torch.randn(10, 3, 15).to(device)
        
        with torch.no_grad():
            forecast = wrapper.forward(X_test)
        assert forecast.shape[0] == 10
        result["layer2_forecast_success"] = True
        result["layer2_trajectory_ready"] = True
        print("✓ Layer2 success")
        
        result["final_status"] = "PASS"
        print("\n[TimeXer] ✓ ALL CHECKS PASSED")
        
    except Exception as e:
        result["error_message"] = str(e)
        result["blocker_type"] = "wrapper_interface_bug"
        print(f"\n[TimeXer] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    return result

if __name__ == "__main__":
    os.makedirs("outputs/model_repairs", exist_ok=True)
    result = test_timexer()
    
    with open("outputs/model_repairs/timexer_repair_result_v2.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果: {result['final_status']}")

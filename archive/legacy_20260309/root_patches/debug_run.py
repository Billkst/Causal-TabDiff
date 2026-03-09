import sys
import traceback

def main():
    try:
        import run_baselines
        # Override sys.argv to simulate running with --debug_mode
        sys.argv = ['run_baselines.py', '--debug_mode']
        run_baselines.main()
    except Exception as e:
        with open('true_error.log', 'w') as f:
            traceback.print_exc(file=f)
        print("Error caught and written to true_error.log")

if __name__ == '__main__':
    main()

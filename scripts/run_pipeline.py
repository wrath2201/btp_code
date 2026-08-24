import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.pipeline import main

if __name__ == "__main__":
    main()

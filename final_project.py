from task1_and_3 import*
import coreg
from coreg import CoxCoRegularized

def main():
    args = sys.argv
    if int(args[1]) == 1 or int(args[1]) == 3:
        create_prediction(args[2], args[3], args[4])
    else:
        coreg.predict_main(cancer=args[2], data_path=args[3], out_path=args[4])

if __name__ == "__main__":
    main()



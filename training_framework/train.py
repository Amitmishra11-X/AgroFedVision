from config import *
from utils import create_folders, set_seed
from dataset import load_dataset

from models.model_registry import build_model
from models.model_info import print_model_statistics

from trainer import ResearchTrainer


def main():

    # -------------------------------------------------
    # Initial Setup
    # -------------------------------------------------

    create_folders()

    set_seed(SEED)

    print("=" * 60)
    print("AgroFedVision Research Training Framework")
    print("=" * 60)

    print("Loading datasets...")

    # -------------------------------------------------
    # Load Dataset
    # -------------------------------------------------

    train_ds, class_names = load_dataset(TRAIN_DIR)

    val_ds, _ = load_dataset(
        VAL_DIR,
        shuffle=False
    )

    test_ds, _ = load_dataset(
        TEST_DIR,
        shuffle=False
    )

    print()

    print("Dataset Loaded Successfully")

    print()

    print("Classes")

    for i, c in enumerate(class_names):

        print(f"{i} -> {c}")

    print()

    print("Total Classes :", len(class_names))

    # -------------------------------------------------
    # Build Model
    # -------------------------------------------------

    print()

    print("Building Model...")

    model = build_model(

        MODEL_NAME,

        len(class_names)

    )

    print_model_statistics(model)

    model.summary()

    # -------------------------------------------------
    # Train Model
    # -------------------------------------------------

    trainer = ResearchTrainer(

        model=model,

        train_ds=train_ds,

        val_ds=val_ds,

        test_ds=test_ds,

        class_names=class_names,

        model_name=MODEL_NAME

) 

    trainer.compile()

    history = trainer.train()

    results = trainer.evaluate()

    trainer.save()

    print()

    print("=" * 60)
    print("Training Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":

    main()
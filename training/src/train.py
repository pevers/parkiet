from dia.model import Dia


def main():
    model = Dia.from_local(
        config_path="config.json",
        checkpoint_path="weights/dia-v0_1.pth",
    )
    print(model)


if __name__ == "__main__":
    main()

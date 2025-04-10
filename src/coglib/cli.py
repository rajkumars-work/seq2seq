import typer

from .utils.api import list, model_info, predict, train, translate
from .version import __version__

app = typer.Typer()


@app.command("version")
def version() -> None:
  print(__version__)


@app.command("list")
def ls() -> None:
  print("models\n------")
  print("\n".join(list()))


@app.command("model")
def model_info_cmd(name: str) -> None:
  print(model_info(name))


@app.command("train")
def train_annotated(
  name: str,
  data: str,
  epochs: int = typer.Option(30, "-e", "--epochs"),
  torch: bool = typer.Option(True, "--torch/--tf"),
) -> None:
  print("train", name, data, epochs, torch)
  j = train(name, data, epochs, torch)
  print(j)


@app.command("translate")
def translate_sentence(name: str, sentence: str) -> None:
  print(translate(name, sentence))


@app.command("predict")
def predict_sentence(name: str, context: str, target: str) -> None:
  print(predict(name, context, target))


def main() -> None:
  app()


if __name__ == "__main__":
  main()

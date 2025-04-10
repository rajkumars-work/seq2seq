import json
from urllib import parse
from urllib.request import urlopen

from .dns_lookup import resolve

Http_Service = "_spotify-cogservice._http.services.gew1.spotify.net"

host = resolve(Http_Service)


def get_host() -> str:
  return resolve(Http_Service)


def base_url() -> str:
  host = resolve(Http_Service)
  return f"http://{host}/v1/cogs/"


def cogserver_get(path: str, host: str = None) -> str:
  host = host if host else get_host()
  url = f"http://{host}/v1/cogs/{path}"
  with urlopen(url) as f:
    resp = json.load(f)
  return resp


def get_model_response(model: str, context: str, prompt: str, host=host) -> str:
  context = parse.quote_plus(context)
  prompt = parse.quote_plus(prompt)
  qs = f"pred?model={model}&context={context}&prefix={prompt}"
  dict = json.loads(cogserver_get(qs, host))
  return dict.get("prediction", "")

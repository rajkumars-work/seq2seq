import requests

API_KEY = "53aeb662cb2ed49b7410c043ea8a3f9f"


def get_weather(location: str) -> str:
  """
  Get the weather for a location

  Args:
    location: The city or state
  """
  w = get_open_weather(location)
  temp = round(w["main"]["temp"] - 273.15, 2)  # kelvin to celsius
  weather = w["weather"][0]["description"]
  return f"{weather} and {temp} celsius"


def get_time(location: str) -> str:
  """
  Get the current time for a location

  Args:
    location: The city or state
  """
  from datetime import datetime, timedelta, timezone

  offset = get_open_weather(location).get("timezone")
  delta = timedelta(seconds=offset)
  time = datetime.now(timezone.utc) - delta
  return time.strftime("%I %M %p")


def get_open_weather(location):
  url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}"
  response = requests.get(url)
  return response.json()


if __name__ == "__main__":
  print(get_weather("New York"))
  print(get_time("Berlin"))

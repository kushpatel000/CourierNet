import paho.mqtt.client as mqtt
import json
from datetime import datetime

# def on_connect(client, userdata, flags, rc):
#     print(f"Connected with result code {rc}")
#     client.subscribe("$SYS/#")
#
# def on_message(client, userdata, msg):
#     print(msg)
#

label = "Fedex"
confidence = 0.9965

time = datetime.now()
timestr = time.strftime("%Y-%m-%d %H:%M:%S")


msg = json.dumps({
    "carrier":label,
    "confidence": f'{confidence*100:6.2f}',
    "timestamp": timestr
})

# msg = f'{label} {confidence*100:6.2f}'



print(msg)
broker_address = "192.168.1.70"
client = mqtt.Client("RPI_Courier")
client.connect(broker_address)
# client.subscribe("Deliveries/Carrier")
client.publish("CourierNet/delivery",payload=msg)

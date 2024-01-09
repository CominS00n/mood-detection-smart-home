from yeelight import Bulb, discover_bulbs

# Discover available Yeelight bulbs on the network
bulbs = discover_bulbs()

# Print information about discovered bulbs
for bulb in bulbs:
    print("ID: {0}, IP: {1}".format(bulb["id"], bulb["ip"]))

# Connect to a specific Yeelight bulb
bulb_ip = "192.168.1.119"  # Replace with the actual IP address of your Yeelight bulb
bulb = Bulb(bulb_ip)

# Turn on the bulb
bulb.turn_on()

# Change the color temperature (optional)
bulb.set_color_temp(4000)

# Change the brightness (optional)
bulb.set_brightness(50)

# Turn off the bulb after a delay (optional)
# import time
# time.sleep(5)  # Wait for 5 seconds
# bulb.turn_off()
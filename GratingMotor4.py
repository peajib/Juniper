

import serial, struct, time, os

PORT = "COM3"
BAUD = 9600
TO   = 0.3

CMD_MOTORTYPE   = 0x02  # 1=linear, 2=rotary
CMD_SET_VELOC   = 0x03  # uint16 (rpm*100)
CMD_CONTINUOUS  = 0x04  # 1=left, 2=right (if supported by your FW; manual may differ)
CMD_STOP        = 0x05  # no params
CMD_DESTINATION = 0x06  # dir(1=right,2=left), pulses (24-bit LE)
CMD_HOME = 0x0B

STATE_FILE = ".romo_abs_pos.txt"  # last commanded absolute

def open_port():
    return serial.Serial(PORT, BAUD, bytesize=8, parity=serial.PARITY_NONE,
                         stopbits=1, timeout=TO, write_timeout=TO)

def p24_le(n): return struct.pack("<I", n)[:3]

def tx(ser, cmd, params=b"", note=""):
    payload = bytes([cmd]) + bytes(params)
    frame   = bytes([0x05, len(payload)]) + payload
    ser.reset_input_buffer(); ser.reset_output_buffer()
    ser.write(frame); ser.flush()
    time.sleep(0.04)
    resp = ser.read(64)
    print(f"{note} TX:{frame.hex(' ')}  RX:{resp.hex(' ')}")
    return resp

def set_motor_type(ser): tx(ser, CMD_MOTORTYPE, b"\x02", "MotorType(rotary)")
def set_velocity(ser, rpm): tx(ser, CMD_SET_VELOC, struct.pack("<H", int(round(rpm*100))), "SetVelocity")
def stop(ser): tx(ser, CMD_STOP, b"", "Stop")

def home(ser, right=True, dwell=0.5, backoff_pulses=0):
    """
    Drive to the mechanical stop using Home(11).
    Param '1' => RIGHT; any other value => LEFT (we use 0x02 for clarity).
    Optionally back off a little after hitting the stop.

    right           : True -> home to RIGHT stop; False -> LEFT stop
    dwell           : time to allow motion towards the stop before issuing STOP
    backoff_pulses  : move away from the stop by this many pulses after homing (0 = none)
    """
    dir_byte = b"\x01" if right else b"\x02"  # per manual: 1=RIGHT, any other=LEFT
    tx(ser, CMD_HOME, dir_byte, f"Home({'R' if right else 'L'})")
    time.sleep(max(0.2, float(dwell)))        # let it reach the mechanical limit
    stop(ser)                                  # clean stop at the limit

    # If you want to move off the hard stop slightly, do a tiny relative move away.
    if backoff_pulses:
        # Define "away" from the stop:
        # If we homed RIGHT, move LEFT; if we homed LEFT, move RIGHT.
        away_is_right = not right
        cur = 0  # we're treating the stop as position 0 in software
        tgt = (backoff_pulses if away_is_right else -backoff_pulses) + cur
        destination(ser, cur, tgt)
        write_software_abs(tgt)
    else:
        # Treat the stop as absolute zero in software
        write_software_abs(0)


def jog(ser, right=True, seconds=0.12):
    # Optional: “re-arm” the drive
    # NOTE: if your manual says 1=right,2=left for Continuous too, swap these two bytes accordingly.
    tx(ser, CMD_CONTINUOUS, b"\x02" if right else b"\x01", f"Jog({'R' if right else 'L'})")
    time.sleep(seconds)
    stop(ser); time.sleep(0.05)

# =============================================================================
# def destination(ser, current_abs, target_abs):
#     delta = target_abs - current_abs
#     # >>> Correct mapping per your manual: 1 = RIGHT, 2 = LEFT
#     dir_byte = b"\x01" if delta >= 0 else b"\x02"
#     params = dir_byte + p24_le(target_abs & 0xFFFFFF)
#     tx(ser, CMD_DESTINATION, params, f"Destination({target_abs} from {current_abs}, delta={delta})")
# =============================================================================

def destination(ser, current_abs, target_abs):
    """
    Move from current_abs -> target_abs.
    Sends |delta| as the 24-bit pulse count and uses dir 1=RIGHT, 2=LEFT.
    Returns the new absolute (target_abs) so caller can update state.
    """
    delta = int(target_abs - current_abs)
    if delta == 0:
        print("Destination: already there.")
        return current_abs

    dir_byte = b"\x01" if delta > 0 else b"\x02"      # 1=RIGHT, 2=LEFT
    pulses  = abs(delta) & 0xFFFFFF                   # controller wants relative pulses
    params  = dir_byte + p24_le(pulses)
    tx(ser, CMD_DESTINATION, params,
       f"Destination({target_abs} from {current_abs}, delta={delta})")
    return target_abs


def read_software_abs():
    try:    return int(open(STATE_FILE).read().strip())
    except: return 0

def write_software_abs(v):
    try: open(STATE_FILE, "w").write(str(v))
    except: pass

if __name__ == "__main__":
    NEW_TARGET = 10000  # change freely between runs

    last_abs = read_software_abs()
    with open_port() as ser:
        stop(ser)                 # clear any lingering motion
        set_motor_type(ser)
        set_velocity(ser, 10.0)
        # jog(ser, right=True)    # optional, if the controller gets “sticky”

        destination(ser, last_abs, NEW_TARGET)

        # give it time to go; don't immediately stop a big move
        time.sleep(1.0)
        destination(ser, last_abs, 1000)

    write_software_abs(NEW_TARGET)
    print(f"Commanded absolute {NEW_TARGET} (previous {last_abs}).")

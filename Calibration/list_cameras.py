"""List all Lucid TRI050S cameras visible on the network.

Filters out non-Lucid devices (other vendors' GigE Vision cameras on the same
network would otherwise show up too). Uses `system.device_infos` so it never
opens a control channel -- pure passive discovery.

Copy the serial numbers it prints into config.py.
"""
from arena_api.system import system


VENDOR = "Lucid Vision Labs"
MODEL_SUBSTRING = "TRI050S"   # change if you ever add a different Lucid model


def main():
    infos = system.device_infos
    matched = [d for d in infos
               if d.get("vendor") == VENDOR and MODEL_SUBSTRING in d.get("model", "")]

    print(f"Discovered {len(infos)} GigE Vision device(s); "
          f"{len(matched)} matching '{VENDOR}' / '{MODEL_SUBSTRING}'.\n")

    if matched:
        header = f"{'Idx':<4} {'Model':<14} {'Serial':<12} {'IP':<16} {'MAC':<19} {'Firmware'}"
        print(header)
        print("-" * len(header))
        for i, d in enumerate(matched):
            print(f"{i:<4} {d['model']:<14} {d['serial']:<12} "
                  f"{d['ip']:<16} {d['mac']:<19} {d['version']}")

    other = [d for d in infos if d not in matched]
    if other:
        print(f"\n(ignored {len(other)} non-Lucid/non-TRI050S device(s):)")
        for d in other:
            print(f"   {d.get('vendor')} {d.get('model')}  ip={d.get('ip')}")


if __name__ == "__main__":
    main()
from autoscript_sdb_microscope_client import SdbMicroscopeClient
import time
import argparse

parser = argparse.ArgumentParser(description='Script to monitor FIB activity and turn beam off if FIB is left on and'
                                             ' not scanning for too long. Prevents the aperture strip from being milled'
                                             ' through since it is positioned in the column prior to the deflector and'
                                             ' the column valves.')
parser.add_argument('--count_limit', type=int, default='240', help='how many counts (30s each) are needed to'
                                                             ' switch beams off. 240 counts is 120 minutes.')
parser.add_argument('--trigger', type=float, default='5.0e-11', help='specimen current (in Ampere) to'
                                                                     ' determining if FIB is active. 50 pA works '
                                                                     'well in our case but may vary on some systems.')
args = parser.parse_args()


def strip_saver(microscope, trigger, limit):
    microscope.connect()
    i = 1
    while i < limit:
        if microscope.beams.ion_beam.is_on is True:
            print("Beam on.")
            #print(microscope.state.specimen_current.value)
            if microscope.state.specimen_current.value < trigger:
                i += 1
                print("Beam blanked. Count is {}".format(i))
                time.sleep(30)
            else:
                i = 1
                print("Beam unblanked.")
                time.sleep(30)

            if i == limit - 1:
                microscope.beams.ion_beam.turn_off()
                microscope.beams.electron_beam.turn_off()

        else:
            print("Beam off.")
            i = 1
            time.sleep(600)


def main():
    microscope = SdbMicroscopeClient()
    limit = args.count_limit
    trigger = args.trigger
    strip_saver(microscope=microscope, trigger=trigger, limit=limit)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import glob
import os
import shutil
import collections

USAGE= """
Writes a new set of '.mdoc' files in which images noted in the corresponding '.adoc' file will be skipped.

USAGE:
%s '<adoc_file_pattern>*' <output_directory>
General, required command-line parameters:
  1. File pattern for '*.adoc' files. An '.mdoc' file will be searched for each '.adoc' file, and a warning will be reported if not found.
  2. Directory where new '.mdoc' files will be written
Outputs:
  A new '.mdoc' file in which skipped images noted in the '.adoc' file denoted with the tag 'setupset.copyarg.skip' will be skipped

General options:
%s '<adoc_file_pattern>*' <output_directory> --mdoc_dir <mdoc_directory> --verbosity
Parameters:
  --mdoc_dir : Directory in which '.mdoc' files can be found
  --verbosity : Increase verbosity

""" % ((__file__,)*2)

MODIFIED= "Modified 2021-01-15"

def main(adoc_file_pattern, outdir, mdoc_dir='.', verbosity=0):
    """
    Writes a new set of '.mdoc' files in which images noted in the corresponding '.adoc' file will be skipped.

    Arguments.
        adoc_file_pattern : File pattern of '*.adoc' files
        outdir : Output directory
        mdoc_dir : Input directory in which '.mdoc' files can be found
        verbosity : Increase verbosity
    """

    if verbosity>=1 : print( "\nVerbosity set to %s\n" % verbosity)

    # Collect MDOC and ADOC files
    if verbosity>=1 : print( "Looking for '.adoc' and '.mdoc' files...")
    mdoc_list, adoc_list= find_adoc_mdoc(adoc_file_pattern, mdoc_dir=mdoc_dir, verbosity=verbosity)
    assert len(mdoc_list) == len(adoc_list), "ERROR!! Number of mdocs %s != Number of adocs %s" % ( len(mdoc_list), len(adoc_list) )

    # Find images to skip, and remove them
    if verbosity>=1 : print( "Looking for skipped images (numbering from 1) and removing them...")
    find_remove_skips(mdoc_list, adoc_list, outdir, verbosity=verbosity)

    if verbosity>=1 : print("\nDone!\n")

def find_adoc_mdoc(adoc_file_pattern, mdoc_dir='.', verbosity=0):
    """
    Finds '.adoc' files and the corresponding '.mdoc' file.

    Arguments.
        adoc_file_pattern : File pattern of '*.adoc' files
        mdoc_dir : Input directory in which '.mdoc' files can be found
        verbosity : Increase verbosity
    """

    adoc_list= sorted(glob.glob(adoc_file_pattern) )
    if verbosity>=1 : print( "Found %s '.adoc' files of the form '%s'" % (len(adoc_list), adoc_file_pattern) )

    mdoc_list= []
    final_adoc_list= []
    couldnt_find= 0

    for adoc_idx, adoc_file in enumerate(adoc_list):
        # ADOC file is assumed to be TOMOGRAM_NAME.mrc/<arbitrary>*.TOMOGRAM_NAME.adoc
        tomogram_name, adoc_basename= os.path.split(adoc_file)

        # Look for MDOC
        mdoc_path= os.path.join(mdoc_dir, tomogram_name + ".mdoc")

        if os.path.exists(mdoc_path):
            mdoc_list.append(mdoc_path)
            final_adoc_list.append(adoc_file)
            if verbosity>=3 : print("  Found '%s'" % mdoc_path)
        else:
            couldnt_find+= 1
            if verbosity>=2 : print("  WARNING! Didn't find '%s'" % mdoc_path)
    # End adoc loop

    if verbosity>=1 :
        mesg= "Found %s '.mdoc' files in '%s/'" % (len(mdoc_list), mdoc_dir)
        if couldnt_find>0 : mesg+= ", but couldn't find %s." % couldnt_find
        mesg+= "\n"
        print(mesg)

    return mdoc_list, final_adoc_list

def find_remove_skips(mdoc_list, adoc_list, outdir, verbosity=0):
    """
    Finds skipped images in '.adoc' files and removes corresponding images in the corresponding '.mdoc' file.

    Arguments.
        mdoc_list : List of mdoc files
        adoc_list : List of adoc files
        adoc_file_pattern : File pattern of '*.adoc' files
        verbosity : Increase verbosity
    """

    # Make output directory
    if not os.path.isdir(outdir) : os.makedirs(outdir)

    # Initialize
    num_tomograms_w_skips= 0

    # Loop through files
    for idx in range( len(mdoc_list) ):
        # Get current files
        old_mdoc_path= mdoc_list[idx]
        adoc_path= adoc_list[idx]

        # New MDOC path
        new_mdoc_path= os.path.join(outdir, os.path.basename(old_mdoc_path) )

        # Parse ADOC file
        skip_list= find_skips_adoc(adoc_path)

        if len(skip_list) > 0 :
            num_tomograms_w_skips+= 1
            if verbosity>=3 : print("  File '%s': Skipping images %s" % (os.path.basename(adoc_path), skip_list) )

            # Parse MDOC file
            mdoc_data_list= parse_mdoc(old_mdoc_path, verbosity=verbosity)

            # Sanity check
            if skip_list[-1] > len(mdoc_data_list):
                print("Uh oh!! Trying to skip image #%s in '%s' which has only %s images" % (skip_list[-1], old_mdoc_path, len(mdoc_data_list) ) )
                exit()

            # Skip list entries
            new_data_list= [elem for idx,elem in enumerate(mdoc_data_list) if idx not in skip_list]
            # (Syntax adapted from https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time)

            # Get list of remaining angles (only needed for printing info to screen)
            if verbosity>=3 : sort_tilt_angles(new_data_list, verbosity=verbosity)

            # Flatten list (Syntax from https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists)
            flat_list= [item for sublist in new_data_list for item in sublist]

            # Write (overwrite!) MDOC file
            with open(new_mdoc_path, "w") as outfile : outfile.write("\n".join(flat_list) )

            if verbosity>=2:
                mesg= "  Wrote '%s' with %s images (removed %s out of %s)" % (
                    new_mdoc_path, len(new_data_list)-1, len(skip_list), len(mdoc_data_list)-1
                    )
        else:
            if verbosity>=3 : print( "  File '%s': Keeping all" % os.path.basename(adoc_path) )

            # Simply copy MDOC file
            shutil.copyfile(old_mdoc_path, new_mdoc_path)

            if verbosity>=2 :
                mdoc_data_list= parse_mdoc(old_mdoc_path, verbosity=verbosity)
                mesg= "  Wrote '%s' with %s images (kept all)" % ( new_mdoc_path, len(mdoc_data_list)-1)
        # End skip if-then

        if verbosity>=3 : mesg+= "\n"
        if verbosity>=2 : print(mesg)
    # End file loop

    if verbosity>=1 : print("Modified %s out of %s files" % (num_tomograms_w_skips, len(mdoc_list) ) )

def find_skips_adoc(adoc_filename, verbose=False):
    """
    Finds skipped images in '.adoc' files.

    Arguments.
        adoc_filename : '.adoc' filename
        verbose : (boolean) Whether to write to screen

    Returns:
        skip_list : List of skipped images
    """

    skip_list= []

    with open(adoc_filename, "r") as batch_file_obj:
        for line_idx, line in enumerate(batch_file_obj):
            if "setupset.copyarg.skip" in line:
                if verbose:
                    print("  File '%s', line %s: '%s'" % (
                        os.path.basename(adoc_filename), line_idx, line.strip()
                    ) )

                skip_range= line.strip().split('=')[1].strip()

                if len(skip_range) > 0:
                    skip_list= parse_range(skip_range)
        # End line loop

    return skip_list

def parse_range(skip_range):
    """
    Deconstructs comma- and hyphen-delineated range into simple list.

    Arguments.
        skip_range : Comma- and hyphen-delineated string

    Returns:
        skip_list : List of skipped images
    """

    skip_list= []

    # Check for commas
    comma_delimited_list= skip_range.split(',')

    for comma_delimited_item in comma_delimited_list:
        # Check for hyphens
        if len( comma_delimited_item.split('-') ) >= 2:
            skip_first= int(comma_delimited_item.split('-')[0])
            skip_last= int(comma_delimited_item.split('-')[1])
            skip_list+= list( range(skip_first, skip_last+1) )
        else:
            try:
                skip_list+= [int(comma_delimited_item)]
            except ValueError:
                print("\nERROR!! Couldn't parse '%s'" % comma_delimited_item)
                print("Exiting...\n")
                exit()
        # End hyphen if-then
    # End comma loop

    return skip_list

def parse_mdoc(mdoc_filename, verbosity=0):
    """
    Parses '.mdoc' file into list, broken down by each image represented in file.

    Arguments.
        mdoc_filename : '.mdoc' filename
        verbosity : Increase verbosity

    Returns:
        sorted_list_of_lists
            List of lists with contents of sorted '.mdoc' file
            Each outer list corresponds to each tilt-series image represented in '.mdoc' file
                The first outer-list element corresponds to the global header
            Each inner-list element represents one line of text
    """

    # Initialize
    unsorted_list= []
    current_data= []

    # Parse MDOC file
    with open(mdoc_filename, "r") as mdoc_file_obj:
        for line_idx, line in enumerate(mdoc_file_obj):
            if "ZValue" in line:
                if verbosity>=5 : print("    Line %s, new image entry: %s" % (line_idx, line.strip() ) )
                unsorted_list.append(current_data)

                # Start new data chunk
                current_data= [line.strip()]
            else:
                current_data.append( line.strip() )

            if verbosity>=6 : print("      Line %s: %s" % (line_idx, line.strip() ) )
        # End line loop

    # Last entry
    unsorted_list.append(current_data)

    # Sort according to angle
    sorted_dict= sort_tilt_angles(unsorted_list, verbosity=verbosity)

    # First entry in final list will be global header, which will preserve numbering starting from 1 for images
    sorted_list_of_lists= [unsorted_list[0]]

    # Now append each image's data having sorted them by angle
    for key, zdata in sorted_dict.iteritems():
        sorted_list_of_lists.append(zdata)

    return sorted_list_of_lists

def sort_tilt_angles(unsorted_list, verbosity=None):
    """
    Sorts lists of lists according to "TiltAngle" entry.

    Arguments.
        unsorted_list
            List of lists with contents of sorted '.mdoc' file
            Each outer list corresponds to each tilt-series image represented in '.mdoc' file
                The first outer-list element corresponds to the global header
            Each inner-list element represents one line of text
        verbosity : Increase verbosity

    Returns:
        sorted_dict : Sorted ordered dictionary with sorted angle as key, and line of text as value
    """

    # Initialize
    zdict= {}

    # Sort entries according to angle
    for zidx, zentry in enumerate(unsorted_list):
        # https://stackoverflow.com/questions/4843158/check-if-a-python-list-item-contains-a-string-inside-another-string
        tilt_string= [s for s in zentry if "TiltAngle" in s]

        # Skip first entry, which has only global variables, not for each image
        if len(tilt_string) != 0:
            tilt_angle= float(tilt_string[0].split('=')[1])
            if verbosity>=4 : print("    img %s: tilt_angle %s" % (zidx, tilt_angle) )
            zdict[tilt_angle]= zentry
    # end zvalue loop

    # Sort according to tilt angle
    sorted_dict= collections.OrderedDict( sorted( zdict.items() ) )

    if verbosity>=3 : print( "  Sorted angles (%s total): %s" % ( len( sorted_dict.keys() ), sorted_dict.keys() ) )

    return sorted_dict

def parse_command_line():
    """
    Parse the command line.  Adapted from sxmask.py

    Arguments:
        None

    Returns:
        parsed arguments object
    """

    parser= argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage=USAGE,
        epilog=MODIFIED
        )

    parser.add_argument(
        "adoc_file_pattern",
        type=str,
        help="File pattern of '*.adoc' files (be sure to include wild card in quotes)")

    parser.add_argument(
        "outdir",
        type=str,
        help="Output directory")

    parser.add_argument("--mdoc_dir",
        type=str,
        default='.',
        help="Directory in which '.mdoc' files can be found")

    parser.add_argument("--verbosity", "-v",
        type=int,
        default=2,
        help="Increase verbosity")
    """
    Verbosity levels:
        0: None
        1: Basic summary
        2: Summary plus warnings
        2: Reports writing of output MDOC files
        3: Plain-text summary of skips from ADOC files
        4: Prints angles before & after skips
        5: Every image entry in MDOC file
        6: Every line in MDOC file
    """

    return parser.parse_args()

if __name__ == "__main__":
    options= parse_command_line()

    ##print args, options  # (Everything is in options.)
    #print(options)
    #exit()

    main(
        options.adoc_file_pattern,
        options.outdir,
        mdoc_dir=options.mdoc_dir,
        verbosity=options.verbosity
        )
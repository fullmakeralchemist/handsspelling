import os
import glob

def rename_images(folder_path):
    # Change the current working directory to the folder with images
    os.chdir(folder_path)

    # Get a list of all image files in the folder
    image_files = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png") + glob.glob("*.gif")

    # Sort the list of image files alphabetically
    image_files.sort()

    # Initialize a counter to create sequential numbers
    counter = 1

    # Rename each image file
    for old_name in image_files:
        # Get the file extension
        extension = os.path.splitext(old_name)[1]

        # Create the new name with the desired format (e.g., "title_1.jpg", "title_2.jpg", etc.)
        new_name = f"thumbsup_{counter}{extension}" #change the word before the _ for the name

        # Rename the file
        os.rename(old_name, new_name)

        # Increment the counter for the next image
        counter += 1

if __name__ == "__main__":
    folder_path = "images\hi"#change path for every folder
    rename_images(folder_path)
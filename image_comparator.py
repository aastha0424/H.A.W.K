import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import load_model, load_image, transform  # Import necessary functions and transformations
import psycopg2
import torch

# Define a threshold for change percentage
CHANGE_THRESHOLD = 0  # Example threshold, adjust based on your requirements

# Establish connection to the database
def connect_db():
    try:
        connection = psycopg2.connect(
            user="postgres",
            password="2004",
            host="localhost",
            port="5432",
            database="drone_imagery"
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Fetch the latest image for a given location
def fetch_latest_image(connection, latitude, longitude):
    try:
        cursor = connection.cursor()
        query = """
        SELECT image_path, timestamp 
        FROM images 
        WHERE latitude = %s AND longitude = %s
        ORDER BY timestamp DESC 
        LIMIT 1;
        """
        cursor.execute(query, (latitude, longitude))
        result = cursor.fetchone()
        cursor.close()
        return result if result else None
    except Exception as e:
        print(f"Error fetching latest image: {e}")
        return None

# Insert new image metadata
def insert_new_image(connection, image_path, latitude, longitude, drone_id):
    try:
        cursor = connection.cursor()
        insert_query = """
        INSERT INTO images (image_path, latitude, longitude, timestamp, drone_id) 
        VALUES (%s, %s, %s, %s, %s);
        """
        timestamp = datetime.datetime.now()  # Current timestamp
        cursor.execute(insert_query, (image_path, latitude, longitude, timestamp, drone_id))
        connection.commit()
        cursor.close()
        print("New image metadata inserted.")
    except Exception as e:
        print(f"Error inserting new image: {e}")

# Insert alert data into the database
def insert_alert(connection, image_id, severity_level, alert_details):
    try:
        cursor = connection.cursor()
        insert_query = """
        INSERT INTO alerts (image_id, alert_timestamp, severity_level, alert_details) 
        VALUES (%s, %s, %s, %s);
        """
        timestamp = datetime.datetime.now()  # Current timestamp
        cursor.execute(insert_query, (image_id, timestamp, severity_level, alert_details))
        connection.commit()
        cursor.close()
        print("Alert inserted.")
    except Exception as e:
        print(f"Error inserting alert: {e}")

# Close the connection
def close_connection(connection):
    if connection:
        connection.close()
        print("Database connection closed.")

# Compare images using the ML model and return change percentage and contours
def compare_images(model, image1_path, image2_path):
    # Load both images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Resize images to the same size if necessary
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Compute absolute difference
    diff = cv2.absdiff(img1, img2)
    # Compute percentage of change
    change_percentage = np.sum(diff) / (img1.size * 255) * 100

    return change_percentage  # Ensure only the percentage is returned
def compare_image(model, image1_path, image2_path, device='cpu'):
    # Load and preprocess images
    t1_image = transform(load_image(image1_path)).unsqueeze(0).to(device)  # Add batch dimension and move to device
    t2_image = transform(load_image(image2_path)).unsqueeze(0).to(device)  # Add batch dimension and move to device
    bi_images = torch.cat([t1_image, t2_image], dim=1)  # [b, tc, h, w]

    # Ensure model and input are on the same device
    model = model.to(device)

    # Make predictions
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predictions = model(bi_images)
        change_prob = predictions['change_prediction']  # [b, 1, h, w]

    # Convert tensor to numpy array and squeeze the batch and channel dimensions
    change_prob_np = change_prob.detach().cpu().numpy().squeeze()  # [h, w]

    # Threshold the change probability map
    change_mask = (change_prob_np > 0.000007).astype(np.uint8)  # Binary mask

    # Detect contours on the change mask
    contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return change_prob_np, contours

# Visualize changes
def visualize_changes(before_image_path, after_image_path, change_percentage):
    # Load the model
    

    # Compare images and get change probability and contours
    change_prob_np, contours = compare_image(model, before_image_path, after_image_path, device='cpu')

    # Load and preprocess images
    t1_image = transform(load_image(before_image_path)).unsqueeze(0)  # Add batch dimension
    t2_image = transform(load_image(after_image_path)).unsqueeze(0)   # Add batch dimension
    t2_image_np = t2_image.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to numpy array

    # Convert the "after" image to uint8 for OpenCV and scale from [0,1] to [0,255]
    t2_image_np_uint8 = (t2_image_np * 255).astype(np.uint8)

    # OpenCV expects BGR, so we need to convert from RGB to BGR
    t2_image_np_bgr = cv2.cvtColor(t2_image_np_uint8, cv2.COLOR_RGB2BGR)

    # Draw red contours around the detected changes
    cv2.drawContours(t2_image_np_bgr, contours, -1, (0, 0, 255), thickness=2)  # Red in BGR is (0, 0, 255)

    # Convert the before image from tensor to numpy array
    t1_image_np = t1_image.squeeze().permute(1, 2, 0).cpu().numpy()  # [h, w, c]

    # Plot the original images and highlighted change areas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Before image
    axes[0].imshow(t1_image_np)
    axes[0].set_title('Before Image')
    axes[0].axis('off')

    # After image
    axes[1].imshow(t2_image_np)
    axes[1].set_title('After Image')
    axes[1].axis('off')

    # Highlighted change areas with red outlines
    axes[2].imshow(cv2.cvtColor(t2_image_np_bgr, cv2.COLOR_BGR2RGB))  # Convert back to RGB for displaying in matplotlib
    axes[2].set_title(f'Highlighted Changes with Outlines')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def generate_alert(connection, image_id, severity, details):
    try:
        cursor = connection.cursor()
        insert_query = """
        INSERT INTO alerts (image_id, alert_timestamp, severity_level, alert_details) 
        VALUES (%s, %s, %s, %s);
        """
        timestamp = datetime.datetime.now()
        cursor.execute(insert_query, (image_id, timestamp, severity, details))
        connection.commit()
        cursor.close()
        print(f"Alert generated for image_id {image_id} with severity {severity}: {details}")
    except Exception as e:
        print(f"Error generating alert: {e}")

# Main execution flow
if __name__ == "__main__":
    # Load the model
    model = load_model()

    # Connect to the database
    conn = connect_db()
    
    if conn:
        # Define latitude and longitude for fetching the before image
        lat, long = 40.7128, -74.0060  # Example coordinates (New York)

        # Fetch the latest image from the database
        result = fetch_latest_image(conn, lat, long)
        if result:
            before_image_path, _ = result
            
            # Define the path for the demo "after" image
            after_image_path = r'C:\codes\Projects\H.A.W.K\after.png'

            # Compare images and get change percentage
            change_percentage = compare_images(model, before_image_path, after_image_path)
            
            # Check if the change percentage exceeds the threshold
            if float(change_percentage) > CHANGE_THRESHOLD:
                print(f"Significant change detected: {change_percentage:.2f}%")
                
                # Insert alert into the database
                generate_alert(conn, 1, 'High', f'Change detected: {change_percentage:.2f}%')  # Assume image_id = 1 for example
            else:
                print(f"No significant change detected: {change_percentage:.2f}%")

            # Visualize changes
            visualize_changes(before_image_path, after_image_path, change_percentage)
        
        else:
            print("No matching coordinates found. Adding new image to the database.")
            # Define the path for the new image
            new_image_path = r'C:\codes\Projects\H.A.W.K\after.png'
            
            # Insert the new image into the database
            insert_new_image(conn, new_image_path, lat, long, "drone_demo")
        
        # Close the connection
        close_connection(conn)

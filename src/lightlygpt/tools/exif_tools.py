# Tools for working with image EXIF data and metadata

import streamlit as st
import sys
import os
from datetime import datetime
from PIL import Image, ExifTags
from ..utils.session_utils import get_active_indices

import warnings

warnings.filterwarnings("ignore", message=".*use_column_width.*")
# Try to import geopy with error handling
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic

    GEOPY_AVAILABLE = True
except ImportError:
    st.warning(
        "geopy package not installed. Location-based filtering will be disabled."
    )
    GEOPY_AVAILABLE = False

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_exif_data(img):
    """Extract EXIF data from an image"""
    exif_data = {}
    try:
        # Get raw EXIF data
        exif = img._getexif()
        if exif:
            # Convert EXIF tags to readable names
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif_data[tag_name] = value
    except Exception:
        pass

    return exif_data


def get_datetime_from_exif(exif_data):
    """Extract datetime from EXIF data"""
    datetime_str = None

    # Try different EXIF tags for datetime
    for tag in ["DateTimeOriginal", "DateTime", "DateTimeDigitized"]:
        if tag in exif_data:
            datetime_str = exif_data[tag]
            break

    if datetime_str:
        try:
            # Parse datetime string (format: '2023:01:01 12:00:00')
            dt = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
            return dt
        except Exception:
            pass

    return None


def get_gps_from_exif(exif_data):
    """Extract GPS coordinates from EXIF data"""
    if "GPSInfo" not in exif_data:
        return None

    try:
        gps_info = exif_data["GPSInfo"]

        # Extract GPS data
        lat_data = gps_info.get(2)
        lat_ref = gps_info.get(1)
        lon_data = gps_info.get(4)
        lon_ref = gps_info.get(3)

        if not lat_data or not lon_data:
            return None

        # Convert degrees, minutes, seconds to decimal degrees
        lat = lat_data[0] + lat_data[1] / 60 + lat_data[2] / 3600
        lon = lon_data[0] + lon_data[1] / 60 + lon_data[2] / 3600

        # Adjust for N/S/E/W
        if lat_ref == "S":
            lat = -lat
        if lon_ref == "W":
            lon = -lon

        return (lat, lon)
    except Exception:
        return None


def filter_by_datetime(date_range_str):
    """
    Filter images by date/time they were taken

    Args:
        date_range_str: Date range in format 'YYYY-MM-DD to YYYY-MM-DD'

    Returns:
        String describing filtered images
    """
    try:
        # Parse date range
        parts = date_range_str.split(" to ")
        if len(parts) != 2:
            return "Invalid date range format. Please use 'YYYY-MM-DD to YYYY-MM-DD'."

        try:
            start_date = datetime.strptime(parts[0].strip(), "%Y-%m-%d")
            end_date = datetime.strptime(parts[1].strip(), "%Y-%m-%d")
            # Set end date to end of day
            end_date = end_date.replace(hour=23, minute=59, second=59)
        except Exception:
            return "Invalid date format. Please use 'YYYY-MM-DD to YYYY-MM-DD'."

        # Use get_active_indices for default behavior
        active_indices = get_active_indices()
        filtered_indices = []
        no_date_indices = []
        for idx in active_indices:
            img_file = st.session_state.uploaded_images[idx]
            img_file.seek(0)
            img = Image.open(img_file)

            # Extract EXIF data
            exif_data = extract_exif_data(img)
            img_date = get_datetime_from_exif(exif_data)

            if img_date:
                if start_date <= img_date <= end_date:
                    filtered_indices.append(idx)
            else:
                no_date_indices.append(idx)

        # Store filtered indices in session state for conversational memory
        st.session_state.last_filtered_indices = filtered_indices

        # Generate response
        if not filtered_indices:
            response = f"No images found taken between {parts[0]} and {parts[1]}."
            if no_date_indices:
                response += f" Note: {len(no_date_indices)} images did not have date information."
            return response

        response = f"Found {len(filtered_indices)} images taken between {parts[0]} and {parts[1]}:\n\n"

        for idx in filtered_indices:
            response += f"Image {idx}: {st.session_state.uploaded_images[idx].name}\n"

        if no_date_indices:
            response += (
                f"\nNote: {len(no_date_indices)} images did not have date information."
            )

        return response

    except Exception as e:
        return f"An error occurred while filtering by date: {str(e)}"


def filter_by_location(location_input):
    """
    Filter images by location

    Args:
        location_input: Either coordinates or a place name

    Returns:
        String describing filtered images
    """
    if not GEOPY_AVAILABLE:
        return "Location filtering requires the geopy package. Please install it with 'pip install geopy'."

    try:
        # Determine if input is coordinates or place name
        is_coords = "," in location_input and all(
            c in "0123456789,.-" for c in location_input.replace(" ", "")
        )

        target_coords = None

        if is_coords:
            # Parse coordinates
            try:
                lat, lon = map(float, location_input.split(","))
                target_coords = (lat, lon)
            except Exception:
                return "Invalid coordinates format. Please use 'latitude, longitude'."
        else:
            # Geocode place name
            try:
                geolocator = Nominatim(user_agent="agent4o_app")
                location = geolocator.geocode(location_input)

                if location:
                    target_coords = (location.latitude, location.longitude)
                else:
                    return f"Could not find location: {location_input}"
            except Exception:
                return "Geocoding failed. Please check your internet connection or try using coordinates."

        # Maximum distance in kilometers
        max_distance = 10

        # Use get_active_indices for default behavior
        active_indices = get_active_indices()
        filtered_indices = []
        no_gps_indices = []
        for idx in active_indices:
            img_file = st.session_state.uploaded_images[idx]
            img_file.seek(0)
            img = Image.open(img_file)

            # Extract EXIF data
            exif_data = extract_exif_data(img)
            img_coords = get_gps_from_exif(exif_data)

            if img_coords:
                distance = geodesic(target_coords, img_coords).kilometers
                if distance <= max_distance:
                    filtered_indices.append((idx, distance))
            else:
                no_gps_indices.append(idx)

        # Sort by distance
        filtered_indices.sort(key=lambda x: x[1])

        # Store filtered indices in session state for conversational memory
        st.session_state.last_filtered_indices = [idx for idx, _ in filtered_indices]

        # Generate response
        if not filtered_indices:
            response = f"No images found near {location_input}."
            if no_gps_indices:
                response += f" Note: {len(no_gps_indices)} images did not have location information."
            return response

        response = f"Found {len(filtered_indices)} images near {location_input}:\n\n"

        for idx, distance in filtered_indices:
            response += f"Image {idx}: {st.session_state.uploaded_images[idx].name} ({distance:.1f} km away)\n"

        if no_gps_indices:
            response += f"\nNote: {len(no_gps_indices)} images did not have location information."

        return response

    except Exception as e:
        return f"An error occurred while filtering by location: {str(e)}"

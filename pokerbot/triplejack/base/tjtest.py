# canny edge detection

import cv2


def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def test(img1, img2):
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt


    # Check if images are loaded correctly
    if img1 is None or img2 is None:
        raise ValueError("One of the images couldn't be loaded. Please check the file paths.")

    # Initialize ORB detector
    orb = cv2.SIFT_create()

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Ensure the descriptors are not None
    if des1 is None or des2 is None:
        raise ValueError("No descriptors found. Please check the images.")

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)

    # Use K-means clustering to group matches
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2).fit(src_pts)
    labels = kmeans.labels_

    # Draw matches for the largest cluster
    largest_cluster = max(set(labels), key=list(labels).count)
    matches = [matches[i] for i in range(len(matches)) if labels[i] == largest_cluster]




    print(f"src_pts: {src_pts}")
    print(f"dst_pts: {dst_pts}")

    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    if M is not None:
        # Get the corners of the template image
        h = img2.shape[0]
        w = img2.shape[1]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Print the transformed points for debugging
        print(f"Transformed points: {dst}")

        # Draw bounding box in the main image
        img1_colored = img1.copy()
        img1_colored = cv2.polylines(img1_colored, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # Draw matches
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=matches_mask,  # draw only inliers
                        flags=2)
        img_matches = cv2.drawMatches(img1_colored, kp1, img2, kp2, matches, None, **draw_params)

        # Show the matches
        
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Homography could not be computed.")

  
 


if __name__ == "__main__":
    image = cv2.imread("/home/generel/Documents/code/python/poker/LetsGoGambling/pokerbot/triplejack/base/tests/test-1720544600.png", cv2.IMREAD_COLOR)
    template = cv2.imread("/home/generel/Documents/code/python/poker/LetsGoGambling/pokerbot/triplejack/base/imgs/hole_heart.png", cv2.IMREAD_COLOR)
    # canny = canny_edge_detection(image)
    # cv2.imshow("canny", canny)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    test(image, template)
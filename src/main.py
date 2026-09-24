from __future__ import print_function
import random
import numpy as np
import cv2
import matplotlib
import argparse
import sys
from PIL import Image

matplotlib.use('module://kitcat') # needed to generate images on Kitty terminal
import matplotlib.pyplot as plt


# plt.rcParams['figure.figsize'] = (10.0, 8.0)
# plt.rcParams['image.interpolation'] = 'nearest'

def fetch_frequencies(low_img, high_img, l_ksize, l_sigma, h_ksize, h_sigma):
    low_pass = cv2.cvtColor(cv2.GaussianBlur(low_img, (l_ksize, l_ksize), l_sigma), cv2.COLOR_BGR2RGB)

    # high_gaus = cv2.GaussianBlur(high_img, (3, 3), 0)
    # high_pass = cv.addWeighted(cv2.convertScaleAbs(cv2.Sobel(high_gaus,1, 0, )), 0.5, cv2.convertScaleAbs(cv2.Sobel()), 0.5);
    high_pass = np.abs(cv2.cvtColor(cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(high_img, (9, 9), h_sigma), cv2.CV_16S, ksize=h_ksize)), cv2.COLOR_GRAY2RGB))
    
    return (low_pass, high_pass)


    # could be useful for alignment
    # high_pass = cv2.cvtColor(cv2.medianBlur(cv2.add(cv2.threshold(cv2.subtract(cv2.add(high_img, 127), cv2.GaussianBlur(high_img, (15, 15), h_sigma)),135,255,cv2.THRESH_TOZERO)[1], 120), 3), cv2.COLOR_GRAY2RGB) 
    # hybrid = cv2.addWeighted(low_pass, (1 - merge_alpha), high_pass, merge_alpha, 0)

def generate_hybrid(low_pass, high_pass, merge_alpha):
    return cv2.addWeighted(low_pass, (1 - merge_alpha), high_pass, merge_alpha, 0)


def harris_detection(img_g):
    img_g = np.float32(img_g)
    out = cv2.cornerHarris(img_g,2,3,0.04)
    img_g[out > 0.01*out.max()] = 255

    return np.uint8(img_g)

def harris_align(high_img_g, low_img_g, out):
    align_img = harris_detection(low_img_g)
    overlay_img = harris_detection(high_img_g)

    orb = cv2.ORB_create()
    
    match_align = cv2.cvtColor(align_img, cv2.COLOR_GRAY2RGB)
    match_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2RGB)
    kp_align, des_align = orb.detectAndCompute(match_align, None)
    kp_overlay, des_overlay = orb.detectAndCompute(match_overlay, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    matches = bf.match(des_align, des_overlay)

    matches = sorted(matches, key = lambda x:x.distance)
    
    average_x_translate = 0
    average_y_translate = 0
    for match in matches:
        x_align, y_align = kp_align[match.queryIdx].pt
        x_overlay, y_overlay = kp_overlay[match.trainIdx].pt
        average_x_translate += x_align - x_overlay
        average_y_translate += y_align - y_overlay
    # print(matches)
    average_x_translate = average_x_translate // len(matches)
    average_y_translate = average_y_translate // len(matches)
    print(average_x_translate,  average_y_translate)

    M = np.float32([[1,0,average_x_translate],[0,1,average_y_translate]])
    
    out = cv2.warpAffine(out, M, (out.shape[1], out.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    return out



        
        

    





# def local_maxima(nums):
    

def middle_out_helper(feature_img):
    rows, cols = feature_img.shape
    x, y = 0, 0


    lower_bound = cols // 10 # start 10% into the picture
    upper_bound = (cols // 10) * 9 # end 90% into the picture

    middle_candidate = (0, 0) # (x, # of hits)
    candidate_y_freq = 0
    for x in range(lower_bound, upper_bound): # iterate through x candidate

        i = 15
        hits = 0
        hit_freq_y = np.zeros(rows)

        # This is the middle-out part
        while x + i < cols and x - i >= 0: # check how many hits there are until one goes out of bounds
            # hits are when x + i and x - i are both equal to hi
            hi = 255
            lo = 120 # Low is 120 because the filter is too unsettling to work on at 4am with a completely black background
            
            
            
            for y in range(rows):
                if feature_img[y, x + i] == hi and feature_img[y, x - i] == hi:
                    hits += 1
                    hit_freq_y[y] = hit_freq_y[y] + 1
            i += 1

        if middle_candidate[1] < hits:
            # print(x)g
            middle_candidate = (x, hits)
            candidate_y_freq = hit_freq_y

    # print(middle_candidate)
    #print(candidate_y_freq)
    # print(cols)

    # used to manually experiment with different y values
    # debug_candidate = np.zeros(candidate_y_freq.shape) 
    #
    # for i in range(debug_candidate.shape[0]):
    #     debug_candidate[i] = 0 if candidate_y_freq[i] == 0 else i
    # print(debug_candidate)
    
    # there could be a bunch of different ways to try and align by y using the frequency's of hits but I think I'll try to just find
    # the biggest "island" of non-zero values and set the y to the middle of the island, ignoring islands outside a bound(since generally eye's are in the middle).
    
    lower_bound = (rows // 10) * 2
    upper_bound = (rows // 10) * 8

    island_coord = (0, 0) # start, end 
    temp_island_len = 0;

    # there might be an off by one error somewhere but the results should hold
    # also since we created a lower  and upper bound I don't think it'll be possible for
    # it to go out of bounds
    for y in range(lower_bound, upper_bound):
        if candidate_y_freq[y] == 0:
            if temp_island_len > island_coord[1] - island_coord[0]:
                island_coord = (y - temp_island_len, y-1)
            temp_island_len = 0
        else:
            temp_island_len += 1

    return (middle_candidate[0], (island_coord[0] + island_coord[1]) // 2)




def middle_out_align(high_img, low_img, out): # Silicon Valley Reference
    ksize = 9
    sigma = 3

    # isolate the facial features
    align_img = cv2.medianBlur(cv2.add(cv2.threshold(cv2.subtract(cv2.add(low_img, 127), cv2.GaussianBlur(low_img, (15, 15), sigma)),135,255,cv2.THRESH_TOZERO)[1], 120), 3) # Accidentally found this filter while experimenting, Noticed that it actually 
                                                                                                                                                                             # keeps a lot of facial features with minimal noise everywhere else
    overlay_img = cv2.medianBlur(cv2.add(cv2.threshold(cv2.subtract(cv2.add(high_img, 127), cv2.GaussianBlur(high_img, (15, 15), sigma)),135,255,cv2.THRESH_TOZERO)[1], 120), 3)


    align_x, align_y = middle_out_helper(align_img)
    overlay_x, overlay_y = middle_out_helper(overlay_img)
    

    M = np.float32([[1,0,(align_x - overlay_x)],[0,1,(align_y - overlay_y)]])

    out = cv2.warpAffine(out, M, (overlay_img.shape[1], overlay_img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    
    # temp = cv2.line(align_img, (align_x, 0), (align_x, align_img.shape[0]), (0,0,0),2)
    # cv2.line(temp, (0, align_y), (align_img.shape[1], align_y), (0,0,0), 2)
    return out


    



def main():
    parser = argparse.ArgumentParser(
        description="A script that takes two images and creates a hybrid image"
    )

    parser.add_argument("low_pass_path", help="relative path to the image in the dataset")
    parser.add_argument("high_pass_path", help="relative path to the image in the dataset")
    parser.add_argument("-a", "--alignment_algorithm", help="harris_align = 0 | middle_out_align = 1", default=0, type=int)
    parser.add_argument("-lk", "--low_kernel_size", help="low pass filter kernel size", default=9, type=int) 
    parser.add_argument("-ls", "--low_pass_sigma", help="low pass filter sigma", default=3, type=int) 
    parser.add_argument("-hk", "--high_kernel_size", help="high pass filter kernel size", default=5, type=int)
    parser.add_argument("-hs", "--high_pass_sigma", help="low pass filter sigma", default=1.4, type=float)
    parser.add_argument("-m", "--merge_alpha", help="strength of high-pass image in percentages 0-1", default=0.5, type=float)

    args = parser.parse_args()
    
    # print(args)

    # Define kernel_size and sigma
    l_ksize = args.low_kernel_size
    l_sigma = args.low_pass_sigma

    h_ksize = args.high_kernel_size
    h_sigma = args.high_pass_sigma

    merge_alpha = args.merge_alpha

    low_img = cv2.imread(args.low_pass_path)
    high_img = cv2.imread(args.high_pass_path)
    low_img_g = cv2.cvtColor(low_img, cv2.COLOR_BGR2GRAY)
    high_img_g = cv2.cvtColor(high_img,cv2.COLOR_BGR2GRAY)

    if args.alignment_algorithm != 0 and args.alignment_algorithm != 1:
        sys.exit()
    feature_img = middle_out_align(high_img_g, low_img_g, high_img_g) if args.alignment_algorithm == 1 else harris_align(high_img_g, low_img_g, high_img_g)
    low_pass, high_pass = fetch_frequencies(low_img, feature_img, l_ksize, l_sigma, h_ksize, h_sigma)

    hybrid = generate_hybrid(low_pass, high_pass, merge_alpha)


    # plt.subplot(1,2,1)
    # plt.imshow(low_pass)
    # plt.title('low')
    # plt.axis('off')
    #
    # plt.subplot(1,2,1)
    # plt.imshow(cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB))
    # plt.title('high')
    # plt.axis('off')

    # feature_img = cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)
    # plt.subplot(1,2,1)
    # plt.imshow(high_img)
    # plt.title('features')
    # plt.axis('off')

    plt.imshow(hybrid)
    plt.title('hybrid')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()

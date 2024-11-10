import math
import networkx as nx
import cv2
import numpy as np



def main():
    img = cv2.imread('images/test_graph7.png')
    _, circles = detect_circles(img.copy(), 30, 50, 20, 10, 30)
    circled_image, adj_matrix = detect_edges(img.copy(), circles, 150, 0.5)

    adj_matrix = np.asarray(adj_matrix)
    G = nx.Graph(adj_matrix)  # this is the undirected Graph created via the adjacency matrix
    # nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black',
    #       font_weight='bold')
    # plt.show()


def detect_circles(img, min_dist, param1, param2, min_radius, max_radius):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur for better performance
    blurred = cv2.GaussianBlur(gray, (5, 5), 20)  # cv2.bilateralFilter(gray,10,50,50)

    # Find circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1,
                               min_dist, param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

    return img, circles



def detect_edges(img, circles, threshold, coverage):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prepare adjacency matrix
    adj_matrix = [[0 for x in range(len(circles[0, :]))] for y in range(len(circles[0, :]))]
    edges = set()

    for i, (cx1, cy1, r1) in enumerate(circles[0, :]):
        for j, (cx2, cy2, r2) in enumerate(circles[0, :]):

            # skip reflexive edge
            if (cx1 == cx2) and (cy1 == cy2): continue

            # draw line in white on black background and get points on the line
            temp_img = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)
            cv2.line(temp_img, (cx1, cy1), (cx2, cy2), 255, 2)
            coords = np.argwhere(temp_img)

            # Calculate distance of current point to the start and endpoint of the retrieved line.
            # If it is closer to the endpoint than the start point this means that the coordinates need to be reversed to obtain he right angles
            dist_to_startpoint = math.sqrt(((coords[0][1] - cx1)) ** 2 + (coords[0][0] - cy1) ** 2)
            dist_to_endpoint = math.sqrt(((coords[-1][1] - cx1)) ** 2 + (coords[-1][0] - cy1) ** 2)
            if dist_to_endpoint < dist_to_startpoint:
                coords = coords[::-1]
            assert math.sqrt(((coords[-1][1] - cx1)) ** 2 + (coords[-1][0] - cy1) ** 2) > math.sqrt(
                ((coords[0][1] - cx1)) ** 2 + (coords[0][
                                                   0] - cy1) ** 2), f'start coords [{coords[0][1]}, {coords[0][0]}]     should [{cx1}, {cy1}]'

            # Check if edge is black (real edge)
            line_pixelvalues = [gray[x[0], x[1]] for x in coords]
            thresholded_pixelvalues = [0 if x < threshold else 1 for x in line_pixelvalues]
            if (sum(thresholded_pixelvalues) < coverage * len(thresholded_pixelvalues)):
                add = True

                # Check other edges starting from the current node for overlapping
                for k, (cx3, cy3, r3) in enumerate(circles[0, :]):
                    # skip current edge and reflexive edge
                    if ((cx1 == cx3) and (cy1 == cy3)) or ((cx3 == cx2) and (cy3 == cy2)): continue

                    # draw line in white on black background and get points on the line
                    temp_img2 = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)
                    cv2.line(temp_img2, (cx1, cy1), (cx3, cy3), 255, 2)
                    coords2 = np.argwhere(temp_img2)

                    # Calculate distance of current point to the start and endpoint of the retrieved line.
                    # If it is closer to the endpoint than the start point this means that the coordinates need to be reversed to obtain he right angles
                    dist_to_startpoint = math.sqrt(((coords2[0][1] - cx1)) ** 2 + (coords2[0][0] - cy1) ** 2)
                    dist_to_endpoint = math.sqrt(((coords2[-1][1] - cx1)) ** 2 + (coords2[-1][0] - cy1) ** 2)
                    if dist_to_endpoint < dist_to_startpoint:
                        coords2 = coords2[::-1]
                    assert math.sqrt(((coords2[-1][1] - cx1)) ** 2 + (coords2[-1][0] - cy1) ** 2) > math.sqrt(
                        ((coords2[0][1] - cx1)) ** 2 + (coords2[0][
                                                            0] - cy1) ** 2), f'start coords [{coords2[0][1]}, {coords2[0][0]}]     should [{cx1}, {cy1}]'

                    # Check if edge is black (real edge)
                    # We only want to check the engles between 'real' edges - otherwise problem occur
                    line_pixelvalues2 = [gray[x[0], x[1]] for x in coords2]
                    thresholded_pixelvalues2 = [0 if x < threshold else 1 for x in line_pixelvalues2]
                    if (sum(thresholded_pixelvalues2) < coverage * len(line_pixelvalues2)):
                        v1 = [(coords[-1][1] - coords[0][1]), (coords[-1][0] - coords[0][0])]
                        v2 = [(coords2[-1][1] - coords2[0][1]), (coords2[-1][0] - coords2[0][0])]

                        # calculate angle
                        v1_u = v1 / np.linalg.norm(v1)
                        v2_u = v2 / np.linalg.norm(v2)
                        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

                        if angle < 0.1:  # 5.7 degree
                            length_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                            length_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

                            # Only allow to add the edge if it is the shorter than all others with angle 0 (below threshold)
                            if length_v1 > length_v2:
                                add = False
                                break

                if add:
                    adj_matrix[i][j] = 1
                    edges.add((i+1, j+1) if i > j else (j+1, i+1))
                    cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)

    return img, adj_matrix, edges


def draw_numbers_on_nodes(img, circles):
    for i, (cx1, cy1, r1) in enumerate(circles[0, :]):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (cx1 + r1, cy1 + r1 // 2)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 4
        # Using cv2.putText() method
        cv2.putText(img, str(i + 1), org, font,
                    fontScale, color, thickness, cv2.LINE_AA)
    return img


def get_closedness_centrality(input_graph, target_node):
    shortest_path_dict = nx.shortest_path_length(input_graph, target_node)
    shortest_path_length = 0
    for _, value in shortest_path_dict.items():
        shortest_path_length += value
    return 0 if shortest_path_length == 0 else 1/shortest_path_length

def get_generalized_closedness_centrality(input_graph):
    closedness_centrality = nx.closeness_centrality(input_graph)
    return closedness_centrality


def get_betweenness_centrality(input_graph):
    betweenness_centrality = nx.betweenness_centrality(input_graph, normalized=False)
    return betweenness_centrality


def get_degree_centrality(input_graph):
    degree_centrality = nx.degree_centrality(input_graph)
    return degree_centrality


def get_eigenvector_centrality(input_graph):
    eigenvector_centrality = nx.eigenvector_centrality(input_graph)
    return eigenvector_centrality


def get_dominance(input_graph, target_node):
    dominated_nodes = set()
    target_neighbors = set(input_graph.neighbors(target_node))
    for node in input_graph.nodes():
        if node != target_node:
            node_neighbors = set(input_graph.neighbors(node))
            if target_node in node_neighbors:
                node_neighbors.remove(target_node)
            if node_neighbors.issubset(target_neighbors):
                dominated_nodes.add(node + 1)
    return dominated_nodes
    # nx.immediate_dominators(input_graph, node)


def get_neighbors(input_graph, target_node):
    neighbors = list(input_graph.neighbors(target_node))
    for i, _ in enumerate(neighbors):
        neighbors[i] += 1
    return neighbors


def get_wiener_index(input_graph):
    directed_G = input_graph.to_directed()
    return nx.wiener_index(directed_G)


def compute_characteristic_pathlength(input_graph):
    wiener_index = get_wiener_index(input_graph)
    n = input_graph.number_of_nodes()

    return wiener_index / (n*(n-1))


def compute_clustering_coef(input_graph):
    n = input_graph.number_of_nodes()

    coefs = []

    for node in range(n):
        neighbors = list(input_graph.neighbors(node))
        H = input_graph.subgraph(neighbors)
        coefs.append(nx.density(H))

    return coefs



def compute_durfee_number(degree_freq):
    for i, degree in enumerate(degree_freq):
        if i > degree:
            return i


def compute_inequalities(degree_freq):
    inequalities = []
    for k in range(1, len(degree_freq) + 1):
        first_sum = sum(degree_freq[0:k])
        second_sum = sum([x if x < k else k for x in degree_freq[k:len(degree_freq)]])
        kk = k * (k - 1)

        inequalities.append([first_sum, second_sum,
                             kk])  # f'k = {k}:  {first_sum} \t<= {kk} + {second_sum} \t {"equal" if first_sum == (kk + second_sum) else ""}'
    return inequalities


def compute_splitance(degree_freq):
    h = compute_durfee_number(degree_freq)

    first_sum = sum(degree_freq[h:len(degree_freq)])
    second_sum = sum(degree_freq[0:h])

    splitance = 0.5 * (h * (h - 1) + first_sum - second_sum)
    return splitance


def compute_thresholdgap(degree_freq):
    h = compute_durfee_number(degree_freq)

    sum = 0
    for i in range(h):
        d_cap = len(list(filter(lambda x: x >= (i + 1) - 1, degree_freq[:i]))) + len(
            list(filter(lambda x: x >= (i + 1), degree_freq[(i + 1):])))
        # print(f'i: {d_cap}')
        sum += abs(d_cap - degree_freq[i])

    return 0.5 * sum


def compute_average_degree(degree_freq):
    return sum(degree_freq) / len(degree_freq)


def compute_number_of_nodes(degree_freq):
    return len(degree_freq)


def compute_number_of_edges(degree_freq):
    return sum(degree_freq) / 2


def compute_density(degree_freq):
    n = len(degree_freq)
    m = sum(degree_freq) / 2

    return (2 * m) / (n * (n - 1))


if __name__ == "__main__":
    main()

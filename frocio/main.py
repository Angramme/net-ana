import streamlit as st
import pandas as pd
import cv2
import numpy as np
from io import StringIO
from analysis import detect_circles
from analysis import *

st.set_page_config(page_title='Graph Analysis', layout='wide')

# Initialize session state
if not bool(st.session_state):
    st.session_state['min_dist'] = 30
    st.session_state['param1'] = 50
    st.session_state['param2'] = 20
    st.session_state['min_radius'] = 10
    st.session_state['max_radius'] = 30

    st.session_state['edge_threshold'] = 150
    st.session_state['edge_coverage'] = 0.5

    st.session_state['graph_image'] = None
    st.session_state['nodes_confirmed'] = False
    st.session_state['edges_confirmed'] = False

    st.session_state['degree_freq'] = '5, 4, 3, 3, 1, 1, 1'


st.title('Graph Analysis')

tab1, tab2 = st.tabs(["Input Image", "Input Degree Frequency"])

# Network image tab
with tab1:

    if st.session_state['graph_image'] is None:
        with st.container():
            uploaded_file = st.file_uploader("Choose a file")

            if uploaded_file is not None:
                # read file
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                st.session_state['graph_image'] = cv2.imdecode(file_bytes, 1)
                st.rerun()

    elif not st.session_state['nodes_confirmed']:
        with st.container():
            col1, col2 = st.columns([2, 1])

            with col1:
                image_with_circles, circles = detect_circles(
                    st.session_state['graph_image'].copy(),
                    st.session_state['min_dist'],
                    st.session_state['param1'],
                    st.session_state['param2'],
                    st.session_state['min_radius'],
                    st.session_state['max_radius']
                )

                st.session_state['circles'] = circles
                st.image(image_with_circles, channels="BGR")
                if st.button('Confirm Nodes'):
                    st.session_state['nodes_confirmed'] = True
                    st.rerun()

            with col2:
                min_dist = st.slider("Minimum distance between circles", 0, 100, key='min_dist', value=st.session_state['min_dist'])
                param1 = 50
                param2 = st.slider("Circle sensitivity (decrease to find more circles)", 0, 100, key='param2', value=st.session_state['param2'])
                min_radius = st.slider("Minimum radius of circles", 0, 100, key='min_radius', value=st.session_state['min_radius'])
                max_radius = st.slider("Maximum radius of circles", 0, 100, key='max_radius', value=st.session_state['max_radius'])

    elif not st.session_state['edges_confirmed']:
        with st.container():
            col1, col2 = st.columns([2, 1])

            with col1:
                image_with_edges, adj_matrix, edges = detect_edges(
                    st.session_state['graph_image'].copy(),
                    st.session_state['circles'],
                    st.session_state['edge_threshold'],
                    st.session_state['edge_coverage'],
                )

                st.session_state['adj_matrix'] = adj_matrix
                st.session_state['edge_list'] = edges
                st.image(image_with_edges, channels="BGR")
                st.subheader(f'Number of edges {sum([sum(x) for x in st.session_state["adj_matrix"]]) / 2}')

                if st.button('Confirm edges'):
                    st.session_state['edges_confirmed'] = True
                    st.rerun()

            with col2:
                edge_threshold = st.slider("Edge threshold (increase if the edges are light)", 0, 255, key='edge_threshold', value=st.session_state['edge_threshold'])
                edge_coverage = st.slider("Edge coverage (increase to accept more edges)", 0.0, 1.0, step=0.01, key='edge_coverage', value=st.session_state['edge_coverage'])

    else:
        adj_matrix = np.asarray(st.session_state['adj_matrix'])
        G = nx.Graph(adj_matrix)
        with st.container():
            st.subheader('Results')
            st.divider()

            col1, col2 = st.columns([1, 1])
            with col1:
                image_with_numbers = draw_numbers_on_nodes(
                    st.session_state['graph_image'].copy(),
                    st.session_state['circles']
                )
                st.image(image_with_numbers, channels="BGR")
            with col2:

                degree_centrality_dict = get_degree_centrality(G)
                closeness_centrality_dict = get_generalized_closedness_centrality(G)
                betweenness_centraility_dict = get_betweenness_centrality(G)
                eigenvector_centrality_dict = get_eigenvector_centrality(G)
                clustering_coef = compute_clustering_coef(G)

                column1_nodes = st.session_state['adj_matrix'].copy()[: len(st.session_state['adj_matrix'])//2]
                column2_nodes = st.session_state['adj_matrix'].copy()[len(st.session_state['adj_matrix'])//2: ]

                col1, col2 = st.columns([1, 1])
                with col1:
                    for i, node in enumerate(column1_nodes):
                        with st.expander(f'Node {i + 1}'):
                            st.markdown(f'''
                                     - **Degree:** {sum(node)}
                                     - **Neighbors** {get_neighbors(G, i)}
                                     - **Degree Centrality:** {degree_centrality_dict[i]}
                                    - **Closeness Centrality:** {get_closedness_centrality(G, i)}
                                     - **Closeness Centrality (Generalized):** {closeness_centrality_dict[i]}
                                     - **Betweenness Centrality:** {betweenness_centraility_dict[i]}
                                     - **Eigenvector Centrality:** {eigenvector_centrality_dict[i]}
                                     - **Dominates** {get_dominance(G, i)}
                                     - **Clustering Coefficient** {clustering_coef[i]}
                                     ''')

                with col2:
                    for i, node in zip([x for x in range(len(column1_nodes), len(st.session_state['adj_matrix']))], column2_nodes):
                        with st.expander(f'Node {i + 1}'):
                            st.markdown(f'''
                                     - **Degree:** {sum(node)}
                                     - **Neighbors** {get_neighbors(G, i)}
                                     - **Degree Centrality:** {degree_centrality_dict[i]}
                                    - **Closeness Centrality:** {get_closedness_centrality(G, i)}
                                     - **Closeness Centrality (Gen.):** {closeness_centrality_dict[i]}
                                     - **Betweenness Centrality:** {betweenness_centraility_dict[i]}
                                     - **Eigenvector Centrality:** {eigenvector_centrality_dict[i]}
                                     - **Dominates** {get_dominance(G, i)}
                                     - **Clustering Coefficient** {clustering_coef[i]}
                                     ''')


        degree_freq = -np.sort(-np.array(st.session_state['adj_matrix']).sum(axis=0)) # sorted

        degree_freq_list = degree_freq.tolist()

        with st.container():
            st.divider()

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.caption('General Numbers')

                st.markdown(f'''
                        - **Number of nodes:** {compute_number_of_nodes(degree_freq)}
                        - **Number of edges:** {compute_number_of_edges(degree_freq)}
                        - **Degree Frequency:** {degree_freq_list}
                        - **Average Degree:** {compute_average_degree(degree_freq)}
                        - **Density:** {compute_density(degree_freq)}
                        - **Wiener Index:** {get_wiener_index(G)}
                        - **Characteristic path length:** {compute_characteristic_pathlength(G)}
                        - **Avg. Clustering Coefficient:** {sum(compute_clustering_coef(G)) / G.number_of_nodes()}

                        ''')

            with col2:
                st.caption('Advanced Numbers')

                st.markdown(f'''
                        - **Durfee number:** {compute_durfee_number(degree_freq)}
                        - **Splitance:** {compute_splitance(degree_freq)}
                        - **Thresholdgap:** {compute_thresholdgap(degree_freq)}
                        ''')
            with col3:
                inequalities = compute_inequalities(degree_freq)
                st.caption('Inequalities')
                for k, inequality in enumerate(inequalities):
                    st.text(
                        f'k = {k + 1}:  {inequality[0]} \t<= {inequality[1]} + {inequality[2]} \t {"equal" if (inequality[0] == (inequality[1] + inequality[2])) else ""}')

        with st.container():
            st.divider()

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.caption('Adjacency matrix')
                df = pd.DataFrame(st.session_state['adj_matrix'], columns=[i+1 for i in range(len(st.session_state['adj_matrix']))])
                df.index += 1
                st.dataframe(df)

            with col2:
                st.caption('Edge List')
                edges = st.session_state['edge_list']
                s = ','.join(map(lambda p: f"({p[0]}, {p[1]})", edges))
                st.code(f"[{s}]")

            with col3:
                pass


# degree frequency tab
with tab2:

    title = st.text_input('Enter a degree frequency', placeholder='5, 3, 3, 2, 1', key='degree_freq')

    if st.session_state['degree_freq'] != '':
        degree_freq = sorted([int(x) for x in st.session_state['degree_freq'].split(",")], reverse=True)

        st.subheader('Results')
        st.divider()

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.caption('General Numbers')

            st.markdown(f'''
            - **Number of nodes:** {compute_number_of_nodes(degree_freq)}
            - **Number of edges:** {compute_number_of_edges(degree_freq)}
            - **Degree Frequency:** {degree_freq}
            - **Average Degree:** {compute_average_degree(degree_freq)}
            - **Density:** {compute_density(degree_freq)}
            ''')

        with col2:
            st.caption('Advanced Numbers')

            st.markdown(f'''
            - **Durfee number:** {compute_durfee_number(degree_freq)}
            - **Splitance:** {compute_splitance(degree_freq)}
            - **Thresholdgap:** {compute_thresholdgap(degree_freq)}
            ''')
        with col3:
            inequalities = compute_inequalities(degree_freq)
            st.caption('Inequalities')

            is_graphic = True
            for k, inequality in enumerate(inequalities):
                if inequality[0] > (inequality[1] + inequality[2]): is_graphic = False

                st.text(f'k = {k+1}:  {inequality[0]} \t<= {inequality[1]} + {inequality[2]} \t {"equal" if (inequality[0] == (inequality[1] + inequality[2])) else ""}')

            if not is_graphic: st.markdown(f'**NOT GRAPHIC**')

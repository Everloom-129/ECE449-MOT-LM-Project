\title{ECE449 Final Project Milestone: \\Referring MOT with Domain Adaptation}
\author{Jie Wang \and Mingchen Li \and Junjie Ren\and Haoxuan Du}

\begin{document}


\section{Introduction}
In the evolving era of artificial intelligence and computer vision, Multmodality models are gaining its importance thanks to the transformer network and incredible GPU performance. Current research concentrates on the the integration of linguistic prompt with visual data for innovative solutions. Our project, "Referring Multi-Object Tracking(MOT) with Domain Adaptation," is positioned at the forefront of this exciting interdisciplinary domain. This project aims to replicate the state-of-the-art(SOTA) research in the realms of natural language processing and multi-object tracking. Furthermore, it is supposed to improve the accuracy and efficiency of object tracking in complex driving scenario.


\section{Literature Review}
According to the project dependency, we mainly focus on the following four articles. Other innovative work like Bytetrack \cite{bytetrack} are also impressive and fundamental to this field, we mainly build our work on the RMOT. 

\subsection{Referring Multi-Object Tracking(RMOT)}
\begin{minipage}{\textwidth}
This paper\cite{rmot} introduces the RMOT architecture, integrating language expressions with multi-object tracking in videos. This novel task addresses existing limitations in single-object tracking. Key contributions include the development of the Refer-KITTI benchmark, the TransRMOT transformer-based architecture, and the use of Higher Order Tracking Accuracy (HOTA) for evaluation. The paper demonstrates TransRMOT's significant improvements over CNN-based counterparts in RMOT task.
\end{minipage}

\begin{itemize}
    \item \textbf{RMOT Task Introduction:} RMOT integrates language expressions with multi-object tracking, addressing single-object focus in existing tasks.
    \item \textbf{Refer-KITTI Benchmark:} A new benchmark derived from the KITTI dataset, featuring 18 videos with 818 expressions, each corresponding to an average of 10.7 objects.
    \item \textbf{TransRMOT Architecture:} A transformer-based architecture utilizing an early-fusion module in its encoder for integrating visual and linguistic features for efficient RMOT.
    \item \textbf{Evaluation Metrics:} Employs Higher Order Tracking Accuracy (HOTA) for benchmark evaluation, focusing on predicted and ground-truth tracklets similarity.
    \item \textbf{Experimental Setup and Training:} Experiments involve training the TransRMOT model with mixed initialization methods, incorporating data augmentation techniques.
    \item \textbf{Method Comparison:} TransRMOT shows significant improvements over CNN-based counterparts in RMOT tasks.
\end{itemize}

\subsection{End-to-End Object Detection with Transformers (DETR)}
\begin{minipage}{\textwidth}
This paper\cite{DBLP} presents a novel approach to object detection, viewing it as a direct set prediction problem. This method simplifies the detection pipeline by eliminating traditional components like non-maximum suppression and anchor generation. DETR employs a transformer-based encoder-decoder architecture, which uses a set-based global loss for unique predictions via bipartite matching and allows for parallel prediction output. It demonstrates competitive accuracy and runtime performance with the established Faster R-CNN on the COCO dataset and extends easily to tasks like panoptic segmentation.
\end{minipage}

\begin{itemize}
    \item \textbf{Novel Approach:} DETR redefines object detection as a direct set prediction task, streamlining the traditional detection pipeline.
    \item \textbf{Transformer Architecture:} Utilizes a transformer encoder-decoder structure, capable of parallel processing and efficient set prediction.
    \item \textbf{Unique Prediction via Bipartite Matching:} Employs a set-based global loss that ensures unique predictions and is invariant to the permutation of predicted objects.
    \item \textbf{Performance:} Shows comparable performance to Faster R-CNN, particularly excelling in detecting large objects.
    \item \textbf{Training and Extensions:} Requires a longer training schedule and benefits from auxiliary decoding losses; adaptable for more complex tasks like panoptic segmentation.
    \item \textbf{Dataset and Technical Details:} Tested on the COCO 2017 detection and panoptic segmentation datasets; trained with AdamW optimizer and ImageNet-pretrained ResNet models.
    \item \textbf{Comparative Analysis:} Offers an extensive comparison with Faster R-CNN, achieving better performance in certain areas while lagging in others.
\end{itemize}

% ⚫ MOTR: https://arxiv.org/abs/2105.03247
\subsection{MOTR: End-to-End Multiple-Object Tracking with Transformer}
\begin{minipage}{\textwidth}
This paper mainly talks about three essential parts\cite{zeng2022motr}. MOTR is a novel end-to-end multiple-object tracking (MOT) framework based on Transformer. It can track objects in video sequences without using any hand-crafted heuristics or post-processing steps. Track query is a key concept introduced by MOTR to model the tracked instances in the entire video. Track query is updated frame-by-frame by interacting with image features and predicts the object trajectory iteratively.
Tracklet-aware label assignment is a method proposed by MOTR to assign track queries to objects in a one-to-one manner. It also handles the cases of newborn and terminated objects by using an entrance and exit mechanism.
\end{minipage}

\begin{itemize}
    \item \textbf{Method:} The paper proposes MOTR, an end-to-end multiple-object tracking framework based on Transformer12. MOTR introduces track query to model the tracked instances in the entire video and performs iterative prediction over time. MOTR also proposes tracklet-aware label assignment, temporal aggregation network, and collective average loss to enhance the temporal modeling and learning.
    \item \textbf{Architecture:} The paper builds MOTR upon Deformable DETR with ResNet50 as the backbone. MOTR consists of a convolutional neural network and a Transformer encoder to extract image features, a Transformer decoder to update the track queries and detect queries, and a query interaction module to handle the object entrance and exit mechanism and perform temporal aggregation.
    \item \textbf{Performance:}The paper evaluates MOTR on three datasets: DanceTrack, MOT17, and BDD100k3. MOTR achieves promising results on DanceTrack, outperforming the state-of-the-art ByteTrack by 6.5 percent on HOTA metric and 8.1 percent on AssA4.
    \item \textbf{Analysis:} The paper conducts ablation studies to demonstrate the effectiveness of different components of MOTR. The paper also discusses the limitations and future directions of MOTR, such as improving the detection performance of newborn objects and enhancing the efficiency of model learning.
\end{itemize}

% ⚫ MOTRv2: https://arxiv.org/abs/2211.09791
\subsection{MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors}
\begin{minipage}{\textwidth}
This paper presents MOTRv2\cite{zhang2023motrv2}, a method for end-to-end multi-object tracking that combines a pretrained object detector (YOLOX) with a modified anchor-based MOTR tracker.
\end{minipage}

\begin{itemize}
    \item \textbf{Proposal query generation:} The paper proposes to use the proposals generated by YOLOX as anchors for the proposal queries, which replace the detect queries in MOTR for detecting new-born objects3. The proposal queries are initialized with a shared query embedding and the confidence score embedding of the proposals.
    \item \textbf{Proposal propagation:} The paper proposes to propagate the proposals from the previous frame to the current frame as anchors for the track queries, which are responsible for tracking the existing objects. The track queries are updated by the self-attention and deformable attention layers in the transformer decoder.
    \item \textbf{Performance:} The paper demonstrates that MOTRv2 achieves significant improvements over the original MOTR and other state-of-the-art methods on three datasets: DanceTrack, BDD100K, and MOT175. The paper also shows the effectiveness of various components of MOTRv2, such as YOLOX proposals, proposal propagation, query denoising, and track query alignment.
\end{itemize}

\section{Methodological Ideas}
%Describe the methods and approaches you plan to employ in your project. This could include experimental setups, data collection strategies, algorithms, etc.

Our immediate focus is to retrain the RMOT on the newly selected training set, followed by rigorous testing on the new test set. Subsequently, we will integrate a language model (potentially RoBERTa or other suitable multi-modal models such as CLIP) to train on the test set. This will be executed in two phases: first, by designing a loss function using only the ground truth of the box (task a); and second, without the ground truth of the box and ID, utilizing pre-training weights of DETR to generate box proposals as pseudo labels (task b). These steps are crucial in refining our RMOT model and ensuring its effectiveness in diverse tracking scenarios.

% Detail the methods and techniques you plan to use.
\subsection{RoBERTa: An optimized method for pretraining self-supervised NLP systems}
RoBERTa\cite{liu2019roberta} is a robustly optimized method for pretraining NLP systems that improves on Bidirectional Encoder Representations from Transformers, or BERT, the self-supervised method released by Google in 2018. As a pre-trained language model, RoBERTa can perform well on some downstream tasks after fine tuning. In RMOT, RoBERTa serves as a feature extractor for language modes and embed the input language query into feature vectors. RoBERTa is just a model of language modes, and we want to replace RoBERTa with a multimodal model, CLIP. Therefore, we can use CLIP's Text Encoder to replace RoBERTa, or we can consider a way to redesign part of the RMOT structure.
% 注意一下上面这一段话的最后一句，我还没完全搞明白怎么缝合CLIP，看看这句话需不需要修改 by LMC

% @RJJ CLIP https://github.com/openai/CLIP
\subsection{CLIP: Contrastive Language-Image Pre-Training}
CLIP \cite{clip} is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.

CLIP can generate semantic embeddings for both the input video frames and language queries in RMOT. By utilizing CLIP as an encoder, we leverage its ability to create meaningful representations for diverse visual and textual information.

CLIP's encoder, being pretrained on a wide range of tasks, can potentially enable zero-shot learning in RMOT. This means the model may generalize well to object tracking scenarios even without specific training data for those scenarios.

\section{Preliminary Experimental Design}
% Outline the experimental design including data sources, experiment setup, and expected outcomes. Also, discuss any progress made so far.

\subsection{Experimental Setup}
% Detail the setup of your experiment, including any equipment or configurations.
% @WJ 
Our team has initialized the experiment by deploying the original Referring Multi-Object Tracking (RMOT) model on a prepaid Linux server, procured from AutoDL (\url{https://www.autodl.com/}). This arrangement facilitates rapid environment setup, efficient code sharing, and collaborative development, with costs equitably shared among our team members.

We have chosen the original KITTI dataset as our primary data source. This dataset includes 'labels\_with\_ids', which are expert annotations provided by the RMOT research team. Given the intricacy involved in modifying the current language feature extractor, our initial approach involves retraining our model on this benchmark dataset to establish a baseline performance.

In terms of language processing capabilities, we have prepared the CLIP model. Our current focus is on developing a methodology to seamlessly integrate CLIP, replacing the original language encoder module in RMOT. This integration aims to enhance the linguistic understanding of our system, potentially improving the accuracy of object tracking in linguistically diverse scenarios.

\subsection{Dataset}
% Discuss the sources of your data, how it will be collected, and any preprocessing steps.
% @LMC KITTI and the new benchmark overview

% Describing the dataset in a natural language
The dataset structure for our RMOT experiments, following the guidelines in the original RMOT paper, is organized as follows:
\begin{verbatim}
|-- refer-kitti
|   |-- KITTI
|           |-- training
|           |-- labels_with_ids
|   |-- expression
\end{verbatim}

The `KITTI` directory contains image data sourced directly from the official KITTI dataset website. For our initial experiments, we have excluded three videos from the KITTI dataset due to their high complexity, which could introduce undue challenges at this early stage.

The `expression` folder contains JSON files with annotations linking object IDs to their respective linguistic descriptions. These IDs correspond to the objects in the `labels\_with\_ids` folder, enabling a direct association between the visual data and their linguistic descriptors. This setup is essential for testing the effectiveness of our integrated CLIP model in interpreting and tracking objects based on linguistic cues.

% Additional information on preprocessing steps or data handling can be added here if necessary.


\section{Conclusion}

This section provides an overview of our project's progress and outlines the future plan of action. We have successfully accomplished several key milestones in our endeavor to enhance Referring Multi-Object Tracking (RMOT) using advanced language models like CLIP.

\subsection{Progress Update}
To date, our team has:
\begin{itemize}
    \item Successfully deployed and run the original RMOT model on our dedicated server.
    \item Executed the CLIP model, which is set to replace the original language encoder in RMOT.
    \item Identified the need to retrain RMOT on a new training set to better adapt it to our project's objectives.
    \item Gained an understanding of how to split the dataset for retraining and testing purposes.
    \item Recognized the necessity to generate new pseudo bounding box (bbox) labels for effective training and testing.
    \item Planned to assess the retrained model's performance to ensure it meets our project's standards.
\end{itemize}

\subsection{Future Plan}
Based on our current progress, we have charted a detailed plan for the period from November 30th to December 24th. The plan is designed to guide our next steps and ensure that we stay on track to achieve our project goals.

\begin{table}[h]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Date Range} & \textbf{Task} \\
\hline
Nov 30 - Dec 7 & Retrain RMOT on new training set \\
\hline
Dec 8 - Dec 15 & Test retrained RMOT on new test set \\
\hline
Dec 16 - Dec 20 & Integrate language model (RoBERTa/CLIP) for task a) \\
\hline
Dec 21 - Dec 24 & Implement and test task b) using DETR for pseudo labels \\
\hline
\end{tabular}
\caption{Scheduled Plan for RMOT Project}
\label{table:1}
\end{table}


\subsection{Closing Remarks}
As we progress, we remain committed to the goal of advancing RMOT through the integration of  language models. We believe that this project will help us get practice on the multi-modality models and gain hands-on experience in the scope of autonomous driving perception. 



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

# V2モデルのみをインポート
from koikoinet_v2 import DiscardModel

def extract_v2_attention(model, x, layer_idx=0):
    """
    Design Intent: V2モデルの月空間変換(Conv2d)後の、指定されたTransformerレイヤーから
    各ヘッドごとの個別アテンションウェイト [num_heads, 12, 12] をダイレクトに抽出する。
    """
    B = x.size(0)
    
    # 1. 月空間へのリシェイプと畳み込み: [B, 24, 48] -> [B, 24, 12, 4] -> [B, nEmb, 12]
    x_space = x.view(B, 24, 12, 4)
    x_feats = model.encoder_block.month_conv(x_space).squeeze(3)
    x_feats = F.relu(x_feats)
    
    # 2. Transformer入力形式への変換: [B, nEmb, 12] -> [B, 12, nEmb]
    x_trans = x_feats.permute(0, 2, 1)
    x_trans = F.layer_norm(x_trans, [x_trans.size(-1)])
    
    # 3. 指定されたレイヤーの直前まで順伝播
    encoder = model.encoder_block.attn_encoder
    for i in range(layer_idx):
        x_trans = encoder.layers[i](x_trans)
        
    # 4. 対象レイヤーの自己アテンションから全ヘッドの生のアテンションウェイトを取得
    target_layer = encoder.layers[layer_idx]
    x_norm = target_layer.norm1(x_trans)
    
    # PyTorch純正のMHAオプション(average_attn_weights=False)でヘッドごとの重みを分離
    _, attn_weights = target_layer.self_attn(
        x_norm, x_norm, x_norm,
        need_weights=True,
        average_attn_weights=False
    )
    
    return attn_weights.squeeze(0) # [num_heads, 12, 12]


def draw_attn_bipartite(labels, attentions, threshold, head_color=0):
    """
    Design Intent: 12ヶ月間のアテンション相関を、左右に同じラベルを並べた二部グラフとして可視化する。
    """
    input_nodes = [lbl + ' ' for lbl in labels]
    output_nodes = [' ' + lbl for lbl in labels]
    
    attn = attentions.detach().cpu().numpy().T
    
    left, right, bottom, top = .1, .9, .1, .9
    v_spacing = (top - bottom) / 12.0
    h_spacing = (right - left)

    G = nx.Graph()
    for i in range(12):
        G.add_node(input_nodes[i], pos=(left + i * v_spacing, left + h_spacing))
        for j in range(12):
            G.add_node(output_nodes[j], pos=(left + j * v_spacing, left))
            if attn[i][j] > threshold:
                G.add_edge(input_nodes[i], output_nodes[j], weight=attn[i][j])

    pos = nx.get_node_attributes(G, 'pos')
    edge_colors = [edge[-1]['weight'] for edge in G.edges(data=True)]

    plt.figure(figsize=(10, 4))
    plt.box(on=None)
    plt.axis('off')
    
    color_map = [plt.cm.Blues, plt.cm.Purples, plt.cm.Oranges, plt.cm.Greens][head_color % 4]
    nx.draw_networkx_nodes(G, pos, node_shape='o', alpha=0)
    
    if edge_colors:
        edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.2, edge_cmap=color_map)
        edges.cmap = color_map

    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(f"Head {head_color} Attention Map (V2 Months Correlation)", fontsize=12)
    return


if __name__ == '__main__':
    # 設定
    target_layer = 0  
    model_path = 'model_v2/discard.pt'
    sample_path = 'dataset_v2/discard/sample.pickle' # 新仕様 [24, 48] のサンプル
    weight_filt_threshold = 0.15 

    # 1. モデルのロード
    model = DiscardModel()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. 新仕様サンプルのロード
    with open(sample_path, 'rb') as f:
        sample = pickle.load(f)
    feature_v2 = sample['feature'].unsqueeze(0) # [1, 24, 48]
    
    # 3. アテンション行列の抽出と可視化
    w_all = extract_v2_attention(model, feature_v2, layer_idx=target_layer)
    month_labels = [f"{m}月" for m in range(1, 13)]
    
    for head in range(4):
        draw_attn_bipartite(month_labels, w_all[head, :, :], weight_filt_threshold, head_color=head)
        
    plt.show()

    # 参考用：マスク有効なカードのQ値スコア出力
    with torch.no_grad():
        output = model(feature_v2).squeeze(0)
        
    print("\n--- 各有効アクションのスコア (参考) ---")
    action_mask = sample['action_mask']
    for idx, is_valid in enumerate(action_mask):
        if is_valid:
            print(f"カード {idx // 4 + 1:2d}月-{idx % 4 + 1}: スコア = {output[idx].item():.2f}")
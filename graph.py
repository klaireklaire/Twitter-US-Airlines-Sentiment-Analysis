import pandas as pd
import matplotlib.pyplot as plt
import six
import numpy as np


# Define the data for the topics
negative_data = {
    "Travel Logistics": ["sit", "waiting", "pass", "seats", "actually", "min", "arrival", "offered", "200", "means"],
    "Food & Service Issues": ["food", "giving", "lga", "little", "issues", "gives", "yesterday", "past", "pilot", "old"],
    "Flight Operations": ["clt", "end", "wow", "having", "flighting", "chicago", "arrive", "bos", "available", "thx"],
    "Customer Service": ["customers", "asked", "stuck", "line", "counter", "attendant", "form", "leaving", "ago", "business"],
    "Operational Efficiency": ["30", "expect", "sleep", "charlotte", "ua", "big", "city", "aa", "lack", "fleets"]
}

positive_data = {
     "Personal/Family Events": ["got", "companion", "iah", "reason", "minutes", "change", "daughter", "bwi", "pretty", "wife"],
    "Celebrations/Intl Travel": ["vegas", "celebrating", "situation", "set", "sharing", "loved", "picture", "international", "great", "think"],
    "Assistance/Communication": ["friday", "news", "omg", "lot", "quick", "base", "video", "assistance", "sending", "yes"],
    "Onboard Exp/Service": ["austin", "onboard", "sleep", "experience", "xoxo", "method", "huge", "problems", "easy", "responding"],
    "General Sentiments/Travel": ["things", "time", "late", "fly", "understand", "sent", "enjoyed", "twitter", "luck", "recovery"]
}


# Convert the dictionary to a DataFrame for better manipulation
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in positive_data.items()]))

# Plotting function to create a table
def render_mpl_table(data, col_width=3.0, row_height=0.325, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax

def main():
    render_mpl_table(df, header_columns=0, col_width=2.5)
    plt.show()

if __name__ == "__main__":
    main()
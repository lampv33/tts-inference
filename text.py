import re


def norm_punc(text):
    text = text.replace('"', '').replace(';', ',').replace(':', '~').replace('(', '~').replace(')', '~').replace('?', '.').replace('!', '.')
    for s in ',.~':
        text = text.replace(s, ' ' + s + ' ')
    text = text.strip(' ,.\n')
    text = re.sub(r'([,.?!~])\s+([,.?!~])', r'\2', text)
    text = ' '.join(text.split())
    return text


def norm_connect_vi(word):
    num_syls = len(word.split('_'))
    if num_syls >= 8:
        word = word.replace('_', ' ')
    return word


def norm_oov(word):
    ws_list = ['s', 'x', 'ph', 'ch', 'g', 'v', 'c', 'p', 'b', 'đ', 'th', 't', 'd', 'k']
    end_ws_remove_list = ['p', 't', 'ch', 'c', 'k', 'd', 'th', 'đ', 'g', 'v']

    for ws in end_ws_remove_list:
        ws = '-' + ws
        if word.endswith(ws):
            word = word[:-len(ws)]
    word_normed = ''
    for s in word.split('-'):
        if s in ws_list:
            s = '<' + s + '>'
        word_normed += s + '-'

    word_normed = word_normed.strip('-')
    return word_normed


def norm_text(text):
    text = norm_punc(text)
    text_norm = ''

    for word in text.split():
        if '_' in word:
            word = norm_connect_vi(word)
        elif '-' in word:
            word = norm_oov(word)

        text_norm += word + ' '

    text_norm = text_norm.strip()
    return text_norm


def split_text_into_sentences(text):
    """
    Split text into sentences based on punctuation marks (., !, ?, newline).
    It combines the punctuation with the preceding text.
    """
    # Split text by sentence-ending punctuation.
    sentence_endings = re.compile(r'([.!?\n])')
    parts = sentence_endings.split(text)
    sentences = []

    # Rejoin parts so that punctuation remains attached to the sentence.
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i].strip() + parts[i + 1]
        sentence = sentence.replace('\n', '.').strip()
        if sentence:
            sentences.append(sentence)

    # If there's any trailing text without a punctuation mark.
    if len(parts) % 2 != 0 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    return sentences


def split_sentence_by_length(sentence, max_len):
    """
    If a sentence is longer than max_len, try to break it into smaller segments.
    First, attempt to split by commas. If a segment still exceeds max_len
    or if no commas are present, fall back to splitting by whitespace.
    """
    if len(sentence) <= max_len:
        return [sentence]

    # Try splitting by comma if available.
    if ',' in sentence:
        segments = []
        current_segment = ""
        parts = [part.strip() for part in sentence.split(',')]
        for part in parts:
            # Prepare candidate by joining the current segment and the next part.
            candidate = current_segment + ", " + part if current_segment else part
            if len(candidate) > max_len:
                # If current_segment is not empty, save it and start a new segment.
                if current_segment:
                    segments.append(current_segment)
                    current_segment = part
                else:
                    # If a single part is too long, fall back to whitespace splitting.
                    segments.extend(split_sentence_by_length(part, max_len))
                    current_segment = ""
            else:
                current_segment = candidate
        if current_segment:
            segments.append(current_segment)

        # Double-check each segment to ensure they are within the limit.
        final_segments = []
        for seg in segments:
            if len(seg) > max_len:
                final_segments.extend(split_sentence_by_length(seg, max_len))
            else:
                final_segments.append(seg)
        return final_segments
    else:
        # No comma found: split by whitespace.
        words = sentence.split()
        segments = []
        current_segment = ""
        for word in words:
            candidate = current_segment + " " + word if current_segment else word
            if len(candidate) > max_len:
                # If the current segment is not empty, finalize it.
                if current_segment:
                    segments.append(current_segment)
                    current_segment = word
                else:
                    # If even one word is too long, we simply add it.
                    segments.append(word)
                    current_segment = ""
            else:
                current_segment = candidate
        if current_segment:
            segments.append(current_segment)
        return segments


def split_text(text, max_len=512):
    """
    Split text into segments that do not exceed max_len characters.

    1. Split the text into sentences.
    2. For sentences that exceed max_len, further split them using split_sentence_by_length.
    3. Concatenate sentences/segments together without exceeding max_len.
    """
    sentences = split_text_into_sentences(text)
    segments = []
    current_segment = ""

    for sentence in sentences:
        # If the sentence itself is too long, split it further.
        if len(sentence) > max_len:
            sub_sentences = split_sentence_by_length(sentence, max_len)
        else:
            sub_sentences = [sentence]

        # Try to aggregate sentences (or sub-sentences) into a segment.
        for sub in sub_sentences:
            candidate = current_segment + " " + sub if current_segment else sub
            if len(candidate) > max_len:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sub
            else:
                current_segment = candidate
    if current_segment:
        segments.append(current_segment.strip())
    return segments


class Text2ID:
    def __init__(self):
        _pad = "$"
        _punctuation = '_,.~;"-():<>?!*[] '
        _letters = 'aáàạãảăắằặẵẳâấầậẫẩbcdđeéèẹẽẻêếềệễểghiíìịĩỉklmnoóòọõỏôốồộỗổơớờợỡởpqrstuúùụũủưứừựữửvxyýỳỵỹỷfjwz'
        symbols = [_pad] + list(_punctuation) + list(_letters)

        self.symbol2index = {symbol: index for index, symbol in enumerate(symbols)}

    def __call__(self, text):
        text = norm_text(text)
        text = '$ ' + text + ' '
   
        indexes = []
        for symbol in text:
            try:
                indexes.append(self.symbol2index[symbol])
            except KeyError:
                print(symbol, text)

        return indexes

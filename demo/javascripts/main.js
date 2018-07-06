let WORD_INDEX = JSON.parse($.ajax({
  dataType: "json",
  url: '../save/tokenizer_word_index.json',
  async: false
}).responseText);

let tokenize = (text) => {
  text = text.toLowerCase()
  text = text.replace(/[^\w\s'-]/gi, '')
  text = text.replace(/\n/g, '').replace(/\t/g, '')
  return text.split(' ')
}

let text_to_sequence = (text) => {
  let tokens = tokenize(text)
  var indicies = tokens.map(x =>
    WORD_INDEX[x] == 'undefined' ? -1 : WORD_INDEX[x])
  return indicies.filter(x => x != -1)
}

let pad_sequence = (seq, maxlen) => {
  if (seq.length > maxlen)
    return seq.slice(seq.length-maxlen, seq.length)
  for (var i = seq.length; i < maxlen; i++)
    seq.unshift(0)
  return seq
}

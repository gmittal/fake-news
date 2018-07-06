WORD_INDEX = JSON.parse('')

let tokenize = (text) => {
  text = text.toLowerCase()
  text = text.replace(/[^\w\s'-]/gi, '')
  text = text.replace(/\n/g, '').replace(/\t/g, '')
  return text.split(' ')
}

let text_to_sequence = (text) => {
  let tokens = tokenize(text)

}
console.log(tokenize('Hello world! this is a great day (sort of), and I want to hear what everyone can\'t here about it'))

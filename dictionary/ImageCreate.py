import sys
from PIL import Image

letters = [Image.open(x) for x in ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png', 'g.png', 'h.png', 'i.png', 'j.png', 'k.png', 'l.png', 'm.png', 'n.png', 'o.png', 
                                  'p.png', 'q.png', 'r.png', 's.png', 't.png', 'u.png', 'v.png', 'w.png', 'x.png', 'y.png', 'z.png', '_space.png']]

print("Type a word or sentence to create braille image.")
word = input().rstrip()

images = []
for letter in word:
    if letter != ' ':
        images.append(letters[ord(letter)-ord('a')])
    else: images.append(letters[26])
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]


word_chunks = word.split(' ')
new_im.save(word_chunks[0]+'.png')

import torch as th
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import BartForConditionalGeneration as BART, BartTokenizer as Tokenizer
import itertools


th.set_printoptions(profile='full')


def process(story):
    story = story.split(' ')
    story = [' '.join(story[i: i + 700]) for i in range(0, len(story), 300)]
    return story


def load_data(tokenizer):
    data = ['test']
    d = 'Stories2/writingPrompts/'
    for name in data:
        with open(d + name + ".wp_target", encoding='utf-8') as f:
            stories = f.readlines()
        # with open(name + ".txt", "w", encoding='utf-8') as o:
        #     for line in stories:
        #         o.write(line.strip() + "\n")
        with open(d + name + '.wp_source', encoding='utf-8') as l:
            prompts = l.readlines()
        # with open(name + "2.txt", "w", encoding='utf-8') as o:
        #     for line in prompts:
        #         o.write(line.strip() + "\n")

        stories = [" ".join(i.split()[0:900]).replace('<newline>', '') for i in stories]
        prompts = [' '.join(i.split()[0:900]).replace('<newline>', '') for i in prompts]

    # with open('Stories/Childrens Stories.txt') as Stories:
    #     lines = [*Stories.readlines()]
    #     prompts = [line for idx, line in enumerate(lines) if idx % 2 == 0]
    #     stories = [line for idx, line in enumerate(lines) if idx % 2 != 0]
    #     stories = [process(story) for story in stories]
    #     for idx in range(len(prompts)):
    #         prompts[idx] = [prompts[idx]] * len(stories[idx])
    #     prompts = list(itertools.chain.from_iterable(prompts))
    #     stories = list(itertools.chain.from_iterable(stories))

    prompts_train, prompts_val, stories_train, stories_val = train_test_split(prompts, stories, shuffle=False, test_size=.2)

    prompts_train = tokenizer(prompts_train, padding='max_length', truncation=True, return_tensors='pt')
    prompts_val = tokenizer(prompts_val, padding='max_length', truncation=True, return_tensors='pt')
    stories_train = tokenizer(stories_train, padding='max_length', truncation=True, return_tensors='pt')
    stories_val = tokenizer(stories_val, padding='max_length', truncation=True, return_tensors='pt')

    return prompts_train, prompts_val, stories_train, stories_val


class Stories(Dataset):
    def __init__(self, text):
        self.prompts, self.stories = text

    def __getitem__(self, idx):
        stories = {key: val[idx] for key, val in self.stories.items()}
        prompts = {key: val[idx] for key, val in self.prompts.items()}
        item = {'input_ids': prompts['input_ids'],
                'attention_mask': prompts['attention_mask'],
                'decoder_input_ids': stories['input_ids'],
                'decoder_attention_mask': stories['attention_mask'],
                'labels': stories['input_ids']}
        return item

    def __len__(self):
        return len(self.prompts.input_ids)


def schedule(step):
    inflection = 400
    if step <= inflection:
        multiplier = step * (5 / inflection)
    else:
        multiplier = (5 * (3600 - (step - inflection))) / 3600

    return multiplier if multiplier >= 1 else 1


def train():
    tokenizer = Tokenizer.from_pretrained('facebook/bart-base')
    prompts_train, prompts_val, stories_train, stories_val = load_data(tokenizer)
    train_dataset = Stories((prompts_train, stories_train))
    print(len(train_dataset))
    val_dataset = Stories((prompts_val, stories_val))

    model = BART.from_pretrained('airKlizz/distilbart-3-3-multi-combine-wiki-news')

    model.cuda()
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(train_dataset, batch_size=1)

    optim = th.optim.AdamW(model.parameters(), lr=4e-5)
    # scheduler = th.optim.lr_scheduler.LambdaLR(optim, lr_lambda=schedule)

    for epoch in range(1):
        print('Epoch', epoch)
        step = 0

        for batch in train_loader:
            optim.zero_grad()

            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            decoder_input_ids = batch['decoder_input_ids'].cuda()
            decoder_attention_mask = batch['decoder_attention_mask'].cuda()
            labels = batch['labels'].cuda()
            labels = th.cat([labels[:, 1:], th.ones((labels.size(0), 1)).long().cuda()], dim=1)

            outputs = model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)
            loss = outputs[0]
            print(step, loss)

            loss.backward()
            optim.step()
            # scheduler.step()
            step += 1

            if step % 250 == 0:
                th.save(model.state_dict(), 'Story Teller.pt')

    # val_losses = []
    # for batch in val_loader:
    #     input_ids = batch['input_ids'].cuda()
    #     attention_mask = batch['attention_mask'].cuda()
    #     decoder_input_ids = batch['decoder_input_ids'].cuda()
    #     decoder_attention_mask = batch['decoder_attention_mask'].cuda()
    #     labels = batch['labels'].cuda()
    #
    #     outputs = model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)
    #     val_losses.append(outputs[0].detach().cpu())
    #
    # print('Val loss', sum(val_losses) / len(val_losses))

    th.save(model.state_dict(), 'Story Teller.pt')


def generate(model, prompt, length):
    model = model.cuda()
    cur_pos = 0
    words_generated = 0
    story = th.LongTensor([[0]]).cuda()
    input_ids = prompt.input_ids[5].view(1, -1).cuda()
    attention_mask = prompt.attention_mask[5].view(1, -1).cuda()
    decoder_input_ids = th.ones((1, 1024), dtype=th.int64).cuda()
    decoder_input_ids[:, 0] = 0

    while words_generated <= length:
        print(words_generated)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )

        # greedy select
        last_hidden_state = outputs[0].detach()

        if cur_pos >= 1023:
            next_token_id = last_hidden_state[:, -1, :].argmax(dim=-1).unsqueeze(0)
            story = th.cat([story, decoder_input_ids[:, 1:514]], dim=1)
            decoder_input_ids = th.cat([decoder_input_ids[:, -511:], next_token_id, th.ones((decoder_input_ids.size(0), 512), dtype=th.int64).cuda()], dim=1)
            decoder_input_ids[:, 0] = 0
            cur_pos = 511
        else:
            # print(last_hidden_state[0, 5, 64], last_hidden_state[0, 5, 0])
            next_token_id = last_hidden_state[:, cur_pos, :].argmax(dim=-1)
            decoder_input_ids[:, cur_pos + 1] = next_token_id
            print(decoder_input_ids)
            cur_pos += 1

        words_generated += 1

    story = th.cat([story, decoder_input_ids[:, 1:]], dim=1).squeeze(0)
    print(story)
    return story


def main():
    # train()
    model = BART.from_pretrained('airKlizz/distilbart-3-3-multi-combine-wiki-news')
    model.load_state_dict(th.load('Story Teller.pt'))
    model.eval()
    tokenizer = Tokenizer.from_pretrained('facebook/bart-base')
    # prompt = ['There was once a young man with nothing in his pocket but a stick and a box that held the moon.']
    # prompt = tokenizer(prompt, padding='max_length', return_tensors='pt')
    prompt, prompts_val, stories_train, stories_val = load_data(tokenizer)
    print(tokenizer.decode(generate(model, prompt, 100), skip_special_tokens=True))

    # prompt = [input('Type a story prompt and hit enter when done:\n')]
    # prompt = tokenizer(prompt, padding='max_length', return_tensors='pt')
    # print('\nHere is the generated story:\n')
    # print(tokenizer.decode(generate(model, prompt, 1200), skip_special_tokens=True))


main()





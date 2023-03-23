use base64::{prelude::BASE64_STANDARD, Engine};
use bstr::ByteSlice;

use rustc_hash::FxHashMap as HashMap;
use crate::CoreBPE;

const ENDOFTEXT: &str = "<|endoftext|>";
const FIM_PREFIX: &str = "<|fim_prefix|>";
const FIM_MIDDLE: &str = "<|fim_middle|>";
const FIM_SUFFIX: &str = "<|fim_suffix|>";
const ENDOFPROMPT: &str = "<|endofprompt|>";

pub fn data_gym_to_mergeable_bpe_ranks(vocab_bpe: &str) -> Option<HashMap<Vec<u8>, usize>> {
    let mut bpe_ranks = HashMap::<Vec<u8>, usize>::default();
    let mut data_gym_byte_to_byte = HashMap::<char, u8>::default();

    for chr in '!'..='~' {
        data_gym_byte_to_byte.insert(chr, chr as u8);
        bpe_ranks.insert(vec![chr as u8], chr as usize);
    }

    for chr in '¡'..='¬' {
        data_gym_byte_to_byte.insert(chr, chr as u8);
        bpe_ranks.insert(vec![chr as u8], chr as usize);
    }

    for chr in '®'..='ÿ' {
        data_gym_byte_to_byte.insert(chr, chr as u8);
        bpe_ranks.insert(vec![chr as u8], chr as usize);
    }

    let mut n = 0;

    for chr in '\0'..=' ' {
        data_gym_byte_to_byte.insert(char::from_u32(n as u32 + 256).unwrap(), chr as u8);
        bpe_ranks.insert(vec![chr as u8], chr as usize);
        n += 1;
    }

    for chr in char::from_u32(127).unwrap()..=char::from_u32(160).unwrap() {
        data_gym_byte_to_byte.insert(char::from_u32(n as u32 + 256).unwrap(), chr as u8);
        bpe_ranks.insert(vec![chr as u8], chr as usize);
        n += 1;
    }

    let del = char::from_u32(173).unwrap();
    data_gym_byte_to_byte.insert(char::from_u32(n as u32 + 256).unwrap(), del as u8);
    bpe_ranks.insert(vec![del as u8], del as usize);

    let mut error = false;
    vocab_bpe
        .split("\n")
        .skip(1)
        .take_while(|line| !line.is_empty())
        .enumerate()
        .map_while(|(index, line)| {
            if line.len() == 0 {
                return None;
            }

            let space_index = line.find(" ");
            if space_index.is_none() {
                error = true;
                println!("No space in: {}", line);
                return None;
            }
            let space_index = space_index.unwrap();

            let mut inner_error = false;
            let key = line[..space_index]
                .chars()
                .map_while(|c| {
                    if data_gym_byte_to_byte.contains_key(&c) {
                        return Some(data_gym_byte_to_byte[&c]);
                    }
                    println!("Missing key for: {} ({})", c, c as u32);
                    error = true;
                    return None;
                })
                .chain(
                    line[space_index + 1..]
                        .chars()
                        .map_while(|c| {
                            if data_gym_byte_to_byte.contains_key(&c) {
                                return Some(data_gym_byte_to_byte[&c]);
                            }
                            inner_error = true;
                            println!("Missing key for: {} ({})", c, c as u32);
                            return None;
                        })
                )
                .collect::<Vec<u8>>();

            if inner_error || error {
                return None;
            }

            bpe_ranks.insert(
                key,
                index + 256
            );

            return Some(());
        })
        .for_each(|_| {});
    
    if error {
        return None;
    }

    return Some(bpe_ranks);
}

pub fn load_tiktoken_bpe(tiktoken_bpe: &str) -> Option<HashMap<Vec<u8>, usize>> {
    let mut error = false;
    let result = tiktoken_bpe
    .split("\n")
    .map_while(|line: &str| {
        if line.is_empty() {
            return None;
        }

        let space_index = line.find(" ");
        if space_index.is_none() {
            error = true;
            return None;
        }
        let space_index = space_index.unwrap();
        let b64 = BASE64_STANDARD.decode(&line[..space_index]).ok();
        if b64.is_none() {
            error = true;
            return None;
        }
        let size = usize::from_str_radix(&line[space_index + 1..], 10).ok();
        if size.is_none() {
            error = true;
            return None;
        }

        return Some((b64.unwrap(), size.unwrap()));
    })
    .collect();

    if error {
        return None;
    }

    return Some(result);
}

async fn get_model(url: &str) -> anyhow::Result<String> {
    Ok(reqwest::get(url).await?.bytes().await?.to_str()?.to_string())
}

pub async fn model_gpt2() -> anyhow::Result<String> {
    get_model("https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe").await
}

pub fn gpt2(model_file: String) -> anyhow::Result<CoreBPE> {
    let mut special_tokens = HashMap::<String, usize>::default();
    special_tokens.insert(ENDOFTEXT.to_string(), 50256);

    return CoreBPE::new(
        data_gym_to_mergeable_bpe_ranks(&model_file).ok_or(anyhow::anyhow!("Failed to load model"))?,
        special_tokens,
        &"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );
}

pub async fn model_r50k_base() -> anyhow::Result<String> {
    get_model("https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken").await
}

pub fn r50k_base(model_file: String) -> anyhow::Result<CoreBPE> {
    let mut special_tokens = HashMap::<String, usize>::default();
    special_tokens.insert(ENDOFTEXT.to_string(), 50256);

    return CoreBPE::new(
        load_tiktoken_bpe(&model_file).ok_or(anyhow::anyhow!("Failed to load model"))?,
        special_tokens,
        &"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );
}

pub async fn model_p50k_base() -> anyhow::Result<String> {
    get_model("https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken").await
}

pub fn p50k_base(model_file: String) -> anyhow::Result<CoreBPE> {
    let mut special_tokens = HashMap::<String, usize>::default();
    special_tokens.insert(ENDOFTEXT.to_string(), 50256);

    return CoreBPE::new(
        load_tiktoken_bpe(&model_file).ok_or(anyhow::anyhow!("Failed to load model"))?,
        special_tokens,
        &"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );
}

pub async fn model_p50k_edit() -> anyhow::Result<String> {
    get_model("https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken").await
}

pub fn p50k_edit(model_file: String) -> anyhow::Result<CoreBPE> {
    let mut special_tokens = HashMap::<String, usize>::default();
    special_tokens.insert(ENDOFTEXT.to_string(), 50256);
    special_tokens.insert(FIM_PREFIX.to_string(), 50281);
    special_tokens.insert(FIM_MIDDLE.to_string(), 50282);
    special_tokens.insert(FIM_SUFFIX.to_string(), 50283);

    return CoreBPE::new(
        load_tiktoken_bpe(&model_file).ok_or(anyhow::anyhow!("Failed to load model"))?,
        special_tokens,
        &"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );
}

pub async fn model_cl100k_base() -> anyhow::Result<String> {
    get_model("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken").await
}

pub fn cl100k_base(model_file: String) -> anyhow::Result<CoreBPE> {
    let mut special_tokens = HashMap::<String, usize>::default();
    special_tokens.insert(ENDOFTEXT.to_string(), 50257);
    special_tokens.insert(FIM_PREFIX.to_string(), 50258);
    special_tokens.insert(FIM_MIDDLE.to_string(), 50259);
    special_tokens.insert(FIM_SUFFIX.to_string(), 50260);
    special_tokens.insert(ENDOFPROMPT.to_string(), 50276);

    return CoreBPE::new(
        load_tiktoken_bpe(&model_file).ok_or(anyhow::anyhow!("Failed to load model"))?,
        special_tokens,
        &"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
    );
}
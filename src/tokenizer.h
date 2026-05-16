#include <ctype.h>
#include <stdint.h>

typedef struct {
  uint32_t vocab_size;
  char **token_table;
  int init_ok;
  int eot_token; // <|endoftext|> token id
} Tokenizer;

extern void safe_printf(const char *piece);
extern void tokenizer_init(Tokenizer *tokenizer, const char *filename);
extern const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id);
extern void tokenizer_free(Tokenizer *tokenizer);
